"""Public IMU training and evaluation entry point for Prism.

The contrastive term uses distances between distinct samples and propagates to
the shared encoder. All unordered non-self pairs in a minibatch are averaged.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import add_data_arguments, encode_labels, fit_label_classes, load_named_split
from models import build_model
from utils.clustering import kmeans


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContrastiveLoss(nn.Module):
    """Half the mean margin loss over all unordered non-self sample pairs.

For P selected pairs, L = sum[u*d^2 + (1-u)*relu(margin-d)^2] / (2*P).
The public implementation defines the contrastive equation's N as pair count,
not the number of samples. This differs from the retained main source's separate
normalization of positive and negative terms.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, features, domains):
        if features.ndim != 2 or domains.ndim != 1 or len(features) != len(domains):
            raise ValueError("Contrastive inputs must have shapes [batch, embedding] and [batch]")
        if len(features) < 2:
            return features.sum() * 0.0
        pairs = torch.triu_indices(len(features), len(features), offset=1, device=features.device)
        distances = torch.pdist(features, p=2)
        positive = domains[pairs[0]].eq(domains[pairs[1]])
        pair_losses = torch.where(positive, distances.square(),
                                  (self.margin - distances).clamp_min(0).square())
        return pair_losses.mean() / 2.0


def classify_embeddings(model, embeddings, domains):
    """Use the same differentiable embeddings for task and contrastive losses."""
    return torch.stack([model.classifiers[int(domain)](embedding)
                        for embedding, domain in zip(embeddings, domains)], dim=0)


def data_loader(split, batch_size, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(split.features),
                            torch.from_numpy(split.labels.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def encode(model, split, batch_size, device):
    model.eval()
    return torch.cat([model(features.to(device)).cpu()
                      for features, _ in data_loader(split, batch_size)], dim=0)


def score(labels, predictions, metadata=None):
    result = {"accuracy": float(accuracy_score(labels, predictions)),
              "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0))}
    if metadata is not None:
        result["by_metadata"] = {}
        for value in np.unique(metadata):
            selected = metadata == value
            result["by_metadata"][str(value)] = score(labels[selected], predictions[selected])
    return result


@torch.no_grad()
def evaluate(model, centers, split, batch_size, device, topk=1):
    model.eval()
    predictions = []
    for features, _ in data_loader(split, batch_size, shuffle=False):
        logits = model.predict(features.to(device), centers.to(device), topk=topk)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return score(split.labels, np.concatenate(predictions), split.metadata)


def pretrain(model, train, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.pretrain_epochs):
        model.train()
        for features, labels in data_loader(train, args.batch_size, shuffle=True):
            features, labels = features.to(device), labels.to(device)
            domains = torch.zeros(len(features), dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(features, domains), labels)
            loss.backward()
            optimizer.step()
    model.copy_first_classifier_params()


def fit(model, train, validation, args, device):
    """Read only train and validation; there is deliberately no test argument."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    contrastive = ContrastiveLoss(args.margin)
    task_loss = nn.CrossEntropyLoss()
    best_f1 = -float("inf")
    best_state, best_centers, best_epoch = None, None, None
    history = []
    centers = None
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        if epoch % args.cluster_interval == 0:
            embeddings = encode(model, train, args.batch_size, device)
            sample_ids = np.random.permutation(len(embeddings))[:1000000]
            centers = kmeans(embeddings[sample_ids].to(device), args.domains,
                             generator=generator).detach()
        model.train()
        sums = np.zeros(3, dtype=np.float64)
        batches = 0
        for features, labels in data_loader(train, args.batch_size, shuffle=True):
            features, labels = features.to(device), labels.to(device)
            embeddings = model(features)
            # Domain membership is a discrete E-step output; detach only its lookup.
            domains = torch.cdist(embeddings.detach(), centers).argmin(dim=1)
            optimizer.zero_grad()
            cl = contrastive(embeddings, domains)
            task = task_loss(classify_embeddings(model, embeddings, domains), labels)
            loss = args.contrastive_weight * cl + task
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            sums += [loss.item(), cl.item(), task.item()]
            batches += 1
        validation_metrics = evaluate(model, centers, validation, args.batch_size, device, args.topk)
        row = {"epoch": epoch + 1, "loss": sums[0] / batches,
               "contrastive_loss": sums[1] / batches, "task_loss": sums[2] / batches,
               "val_accuracy": validation_metrics["accuracy"],
               "val_macro_f1": validation_metrics["macro_f1"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        # Include epoch 1; the source could leave model_best undefined if it never improved.
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            best_centers = centers.detach().cpu().clone()
            best_epoch = epoch + 1
    model.load_state_dict(best_state)
    return best_centers, best_epoch, best_f1, history


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--implementation", choices=["corrected"], default="corrected",
                        help="The public CLI uses the corrected implementation")
    parser.add_argument("--backbone", choices=["FCN", "ConvLSTM", "Transformer"], default="FCN")
    parser.add_argument("--domains", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--cluster-interval", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N; select an available GPU explicitly")
    parser.add_argument("--export-pretrain-embeddings",
                        help="Optional .npy path for training embeddings after M0 pretraining")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if min(args.domains, args.epochs, args.batch_size, args.cluster_interval, args.topk) < 1:
        parser.error("domains, epochs, batch size, cluster interval, and topk must be positive")
    if args.pretrain_epochs < 0 or args.lr <= 0 or args.contrastive_weight < 0 or args.margin < 0:
        parser.error("Invalid training hyperparameter")
    if args.topk > args.domains:
        parser.error("topk must not exceed domains")
    for suffix in ("features", "labels"):
        paths = [Path(getattr(args, split + "_" + suffix)).resolve()
                 for split in ("train", "val", "test")]
        if len(set(paths)) != 3:
            parser.error("train/val/test %s must be distinct files" % suffix)
    return args


def run(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    train = load_named_split(args, "train")
    classes = fit_label_classes(train.labels)
    train.labels = encode_labels(train.labels, classes)
    validation = load_named_split(args, "val", classes)
    if train.features.shape[1:] != validation.features.shape[1:]:
        raise ValueError("Train and validation window shapes differ")
    if args.domains > len(train.features):
        raise ValueError("domains exceeds training sample count")
    model = build_model(args.backbone, train.features.shape[2], train.features.shape[1],
                        len(classes), args.domains).to(device)
    print("Implementation: corrected; contrastive gradient: enabled; all unordered non-self pairs")
    pretrain(model, train, args, device)
    if args.export_pretrain_embeddings:
        destination = Path(args.export_pretrain_embeddings)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, encode(model, train, args.batch_size, device).numpy())
        np.save(destination.with_name(destination.stem + "_labels.npy"), train.labels)
    centers, epoch, best_f1, history = fit(model, train, validation, args, device)
    # Test data is first opened after all model/epoch selection has finished.
    test = load_named_split(args, "test", classes)
    if train.features.shape[1:] != test.features.shape[1:]:
        raise ValueError("Train and test window shapes differ")
    test_metrics = evaluate(model, centers, test, args.batch_size, device, args.topk)
    model_config = {"name": args.backbone, "channels": train.features.shape[2],
                    "length": train.features.shape[1], "classes": len(classes),
                    "domains": args.domains}
    torch.save({"state_dict": model.cpu().state_dict(), "centers": centers,
                "label_classes": classes.tolist(), "model_config": model_config,
                "normalization": args.normalization, "best_epoch": epoch,
                "implementation": "corrected",
                "contrastive_gradient": "enabled",
                "contrastive_reduction": "mean_over_unordered_nonself_pairs"},
               output / "model.pt")
    summary = {"implementation": args.implementation,
               "contrastive_gradient": "enabled",
               "contrastive_reduction": "mean_over_unordered_nonself_pairs",
               "best_epoch_by_validation_f1": epoch, "best_validation_macro_f1": best_f1,
               "test_evaluation_count": 1, "test": test_metrics,
               "model_config": model_config, "arguments": vars(args)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return summary


if __name__ == "__main__":
    run(get_args())
