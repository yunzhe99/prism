"""Three Prism backbone architectures based on the authors' source snapshot.

Layers and widths follow that source. FCN output size is computed from input
length instead of dataset-specific constants. Zero-distance routing is bounded.
ConvLSTM execution is rewritten; legacy checkpoint keys require explicit mapping.
"""
from types import SimpleNamespace
import torch
from torch import nn
from .attention import Seq_Transformer

class FCN_prism(nn.Module):
    def __init__(self, args, n_channels, seq_length, n_classes, out_channels=128):
        super(FCN_prism, self).__init__()

        # Create convolution blocks
        self.conv_block1 = self.create_conv_block(n_channels, 32)
        self.conv_block2 = self.create_conv_block(32, 64)
        self.conv_block3 = self.create_conv_block(64, out_channels)

        self.out_len = seq_length
        for _ in range(3):
            # Conv1d(k=8,p=4,s=1), then MaxPool1d(k=2,p=1,s=2).
            self.out_len = (self.out_len + 3) // 2

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels
        
        # Create classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(self.out_len * out_channels, n_classes) for _ in range(args.cluster_num)
        ])

    def create_conv_block(self, in_channels, out_channels):
        """Create a convolution block with the given input and output channels."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35) if out_channels == 32 else nn.Identity()
        )

    def forward(self, x_in, cluster_index=[-1]):
        """Forward pass through the network."""
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = x.reshape(x.shape[0], -1)

        if cluster_index[0] == -1:
            return x
        else:
            return torch.stack([self.classifiers[int(cluster_each.item())](x[index, :].unsqueeze(0))[0] for index, cluster_each in enumerate(cluster_index)], dim=0)
        
    def predict(self, x, clustering_centers, topk=1, weights=None):
        """Predict the class for the given input and clustering centers."""
        embeddings = self.forward(x)
        distances = torch.cdist(embeddings, clustering_centers)
        _, topk_indices = torch.topk(distances, k=min(topk, len(self.classifiers)), dim=1, largest=False)

        if weights is None:
            weights = 1.0 / distances.gather(1, topk_indices).clamp_min(1e-8)

        predictions = []
        for i, indices in enumerate(topk_indices):
            predictions.append(torch.stack([self.classifiers[int(index)](embeddings[i, :].unsqueeze(0))[0] for index in indices], dim=0))
        predictions = torch.stack(predictions, dim=0)
        # Apply weights to the predictions and sum them up.
        weighted_predictions = []
        for prediction_index, prediction in enumerate(predictions):
            weighted_prediction = 0
            weight_item = weights[prediction_index]
            for weight_index, weight in enumerate(weight_item):
                weighted_prediction += weight.item() * prediction[weight_index]
            weighted_predictions.append(weighted_prediction)
        weighted_predictions = torch.stack(weighted_predictions, dim=0)

        return weighted_predictions
    

    def copy_first_classifier_params(self):
        """Copy the parameters of the first classifier to all other classifiers in the list."""
        first_classifier_params = [param.clone() for param in self.classifiers[0].parameters()]

        for classifier in self.classifiers[1:]:
            for param, ref_param in zip(classifier.parameters(), first_classifier_params):
                param.data.copy_(ref_param.data)

class DeepConvLSTM_prism(nn.Module):
    def __init__(self, args, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128):
        """Implement the four-convolution/two-LSTM architecture with batch-first tensors.

        This release implementation uses a convolution stack rather than the
        retained CL-HAR-derived execution code. Legacy checkpoint keys differ.
        """
        super().__init__()
        convolution_layers = []
        input_width = 1
        for _ in range(4):
            convolution_layers.extend((
                nn.Conv2d(input_width, conv_kernels, kernel_size=(kernel_size, 1)),
                nn.ReLU(),
            ))
            input_width = conv_kernels
        self.temporal_convolutions = nn.Sequential(*convolution_layers)
        self.sequence_dropout = nn.Dropout(p=0.5)
        self.sequence_model = nn.LSTM(
            input_size=n_channels * conv_kernels,
            hidden_size=LSTM_units,
            num_layers=2,
            batch_first=True,
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(LSTM_units, n_classes) for _ in range(args.cluster_num)
        )

    def forward(self, x, cluster_index=[-1]):
        # [batch, time, sensor] -> [batch, filter, time, sensor].
        activations = self.temporal_convolutions(x[:, None, :, :])
        # Flatten sensor then filter, retaining the architecture's feature order.
        sequence = activations.movedim(1, -1).flatten(start_dim=2)
        encoded_sequence, _ = self.sequence_model(self.sequence_dropout(sequence))
        embedding = encoded_sequence.select(dim=1, index=encoded_sequence.size(1) - 1)
        if int(cluster_index[0]) == -1:
            return embedding
        logits = [
            self.classifiers[int(domain)](row)
            for row, domain in zip(embedding, cluster_index)
        ]
        return torch.stack(logits)
        
        
    def predict(self, x, clustering_centers, topk=1, weights=None):
        """Predict the class for the given input and clustering centers."""
        embeddings = self.forward(x)
        distances = torch.cdist(embeddings, clustering_centers)
        _, topk_indices = torch.topk(distances, k=min(topk, len(self.classifiers)), dim=1, largest=False)

        if weights is None:
            weights = 1.0 / distances.gather(1, topk_indices).clamp_min(1e-8)

        predictions = []
        for i, indices in enumerate(topk_indices):
            predictions.append(torch.stack([self.classifiers[int(index)](embeddings[i, :].unsqueeze(0))[0] for index in indices], dim=0))
        predictions = torch.stack(predictions, dim=0)
        # Apply weights to the predictions and sum them up.
        weighted_predictions = []
        for prediction_index, prediction in enumerate(predictions):
            weighted_prediction = 0
            weight_item = weights[prediction_index]
            for weight_index, weight in enumerate(weight_item):
                weighted_prediction += weight.item() * prediction[weight_index]
            weighted_predictions.append(weighted_prediction)
        weighted_predictions = torch.stack(weighted_predictions, dim=0)

        return weighted_predictions
    
    def copy_first_classifier_params(self):
        """Copy the parameters of the first classifier to all other classifiers in the list."""
        first_classifier_params = [param.clone() for param in self.classifiers[0].parameters()]

        for classifier in self.classifiers[1:]:
            for param, ref_param in zip(classifier.parameters(), first_classifier_params):
                param.data.copy_(ref_param.data)

class Transformer_prism(nn.Module):
    def __init__(self, args, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1):
        super(Transformer_prism, self).__init__()

        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        # Create classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(dim, n_classes) for _ in range(args.cluster_num)
        ])

    def forward(self, x, cluster_index=[-1]):
        x = self.transformer(x)
        if cluster_index[0] == -1:
            return x
        else:
            return torch.stack([self.classifiers[int(cluster_each.item())](x[index, :].unsqueeze(0))[0] for index, cluster_each in enumerate(cluster_index)], dim=0)

    def predict(self, x, clustering_centers, topk=1, weights=None):
        """Predict the class for the given input and clustering centers."""
        embeddings = self.forward(x)
        distances = torch.cdist(embeddings, clustering_centers)
        _, topk_indices = torch.topk(distances, k=min(topk, len(self.classifiers)), dim=1, largest=False)

        if weights is None:
            weights = 1.0 / distances.gather(1, topk_indices).clamp_min(1e-8)

        predictions = []
        for i, indices in enumerate(topk_indices):
            predictions.append(torch.stack([self.classifiers[int(index)](embeddings[i, :].unsqueeze(0))[0] for index in indices], dim=0))
        predictions = torch.stack(predictions, dim=0)
        # Apply weights to the predictions and sum them up.
        weighted_predictions = []
        for prediction_index, prediction in enumerate(predictions):
            weighted_prediction = 0
            weight_item = weights[prediction_index]
            for weight_index, weight in enumerate(weight_item):
                weighted_prediction += weight.item() * prediction[weight_index]
            weighted_predictions.append(weighted_prediction)
        weighted_predictions = torch.stack(weighted_predictions, dim=0)

        return weighted_predictions

    def copy_first_classifier_params(self):
        """Copy the parameters of the first classifier to all other classifiers in the list."""
        first_classifier_params = [param.clone() for param in self.classifiers[0].parameters()]

        for classifier in self.classifiers[1:]:
            for param, ref_param in zip(classifier.parameters(), first_classifier_params):
                param.data.copy_(ref_param.data)


def build_model(name, channels, length, classes, domains):
    if domains < 1 or classes < 1:
        raise ValueError("domains and classes must be positive")
    args = SimpleNamespace(cluster_num=domains)
    if name == "FCN":
        return FCN_prism(args, channels, length, classes)
    if name == "ConvLSTM":
        if length < 17:
            raise ValueError("ConvLSTM requires at least 17 time steps")
        return DeepConvLSTM_prism(args, channels, classes)
    if name == "Transformer":
        return Transformer_prism(args, channels, length, classes)
    raise ValueError("Unsupported backbone: " + name)
