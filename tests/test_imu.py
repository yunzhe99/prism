import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from data_utils import encode_labels, fit_label_classes, load_split, select_column

try:
    import torch
    import prism
    from models import build_model
    from utils.clustering import kmeans
except ImportError:
    torch = None


class DataTests(unittest.TestCase):
    def test_label_and_metadata_columns_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            features = np.ones((4, 24, 6), dtype=np.float32)
            labels = np.repeat(np.array([[3, 10], [7, 11], [3, 12], [7, 13]])[:, None], 24, axis=1)
            np.save(directory / "x.npy", features)
            np.save(directory / "y.npy", labels)
            split = load_split(directory / "x.npy", directory / "y.npy",
                               label_index=0, metadata_index=1)
            np.testing.assert_array_equal(split.labels, [3, 7, 3, 7])
            np.testing.assert_array_equal(split.metadata, [10, 11, 12, 13])
            np.testing.assert_array_equal(np.load(directory / "y.npy"), labels)

    def test_explicit_selection_and_consistent_windows(self):
        with self.assertRaises(ValueError):
            select_column(np.zeros((3, 2)))
        labels = np.zeros((3, 4, 2))
        labels[0, 1, 0] = 1
        with self.assertRaises(ValueError):
            select_column(labels, 0)

    def test_unseen_classes_cannot_change_training_mapping(self):
        classes = fit_label_classes(np.array([5, 2, 5]))
        np.testing.assert_array_equal(encode_labels(np.array([5, 2]), classes), [1, 0])
        with self.assertRaises(ValueError):
            encode_labels(np.array([99]), classes)


@unittest.skipIf(torch is None, "Install requirements-imu.txt for torch checks")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_formula_and_contrastive_only_encoder_step(self):
        # A linear encoder turns one-hot samples into h = [0, 1, 3].
        # M=4 gives per-pair half-losses [0.5, 0.5, 2], mean 1.
        encoder = torch.nn.Linear(3, 1, bias=False).double()
        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[0., 1., 3.]], dtype=torch.float64))
        inputs = torch.eye(3, dtype=torch.float64)
        domains = torch.tensor([0, 0, 1])
        objective = prism.ContrastiveLoss(margin=4.)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
        before = encoder(inputs)
        loss = objective(before, domains)
        self.assertAlmostEqual(loss.item(), 1., places=12)
        loss.backward()
        torch.testing.assert_close(encoder.weight.grad,
                                   torch.tensor([[0., 1., -1.]], dtype=torch.float64))
        optimizer.step()  # No task loss participates in this update.
        after = encoder(inputs)
        torch.testing.assert_close(after, torch.tensor([[0.], [0.9], [3.1]], dtype=torch.float64))
        old_distances, new_distances = torch.pdist(before), torch.pdist(after)
        self.assertLess(new_distances[0].item(), old_distances[0].item())  # same domain
        self.assertGreater(new_distances[1].item(), old_distances[1].item())
        self.assertGreater(new_distances[2].item(), old_distances[2].item())
        self.assertAlmostEqual(objective(after, domains).item(), 0.81, places=12)

    def test_contrastive_only_updates_actual_fcn_encoder_not_heads(self):
        torch.manual_seed(13)
        model = build_model("FCN", 6, 24, 2, 2).eval()
        inputs = torch.randn(4, 24, 6)
        domains = torch.tensor([0, 0, 1, 1])
        before = model.conv_block1[0].weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss = prism.ContrastiveLoss(margin=4.)(model(inputs), domains)
        loss.backward()
        gradient = model.conv_block1[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(gradient.abs().sum().item(), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.classifiers.parameters()))
        optimizer.step()
        self.assertGreater((model.conv_block1[0].weight - before).abs().sum().item(), 0.)

    def test_singleton_contrastive_zero_keeps_gradient_path(self):
        embedding = torch.randn(1, 5, requires_grad=True)
        loss = prism.ContrastiveLoss()(embedding, torch.tensor([0]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(embedding.grad, torch.zeros_like(embedding))

    def test_duplicate_embeddings_and_singletons_are_finite(self):
        centers = kmeans(torch.ones(4, 3), 2, max_iterations=3)
        self.assertTrue(torch.isfinite(centers).all().item())
        centers = kmeans(torch.ones(1, 3), 1)
        self.assertEqual(tuple(centers.shape), (1, 3))

    def test_three_retained_backbones_and_zero_distance_routing(self):
        for name in ("FCN", "ConvLSTM", "Transformer"):
            model = build_model(name, 6, 24, 2, 2).eval()
            inputs = torch.randn(3, 24, 6)
            with torch.no_grad():
                embeddings = model(inputs)
                predictions = model.predict(inputs, embeddings[:2], topk=1)
            self.assertEqual(tuple(predictions.shape), (3, 2))
            self.assertTrue(torch.isfinite(predictions).all().item())

    def test_validation_selects_first_epoch_and_test_is_opened_after_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rng = np.random.RandomState(0)
            for split in ("train", "val", "test"):
                np.save(directory / (split + "_x.npy"), rng.randn(8, 24, 6).astype(np.float32))
                np.save(directory / (split + "_y.npy"), np.arange(8) % 2)
            argv = ["--dataset", "uci", "--normalization", "none", "--domains", "2",
                    "--pretrain-epochs", "0", "--epochs", "2",
                    "--batch-size", "8", "--output", str(directory / "out")]
            for split in ("train", "val", "test"):
                argv += ["--" + split + "-features", str(directory / (split + "_x.npy")),
                         "--" + split + "-labels", str(directory / (split + "_y.npy"))]
            args = prism.get_args(argv)
            self.assertEqual(args.implementation, "corrected")
            events = []
            real_load = prism.load_named_split
            real_fit = prism.fit
            real_evaluate = prism.evaluate
            evaluation_count = [0]

            def load_spy(arguments, name, classes=None):
                events.append("load_" + name)
                return real_load(arguments, name, classes)

            def evaluate_spy(*a, **kw):
                result = real_evaluate(*a, **kw)
                evaluation_count[0] += 1
                # Force validation to decline after epoch 1.
                if evaluation_count[0] <= 2:
                    result["macro_f1"] = [0.8, 0.2][evaluation_count[0] - 1]
                return result

            def fit_spy(*a, **kw):
                result = real_fit(*a, **kw)
                events.append("fit_finished")
                return result

            with mock.patch.object(prism, "load_named_split", side_effect=load_spy), \
                    mock.patch.object(prism, "fit", side_effect=fit_spy), \
                    mock.patch.object(prism, "evaluate", side_effect=evaluate_spy), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = prism.run(args)
            self.assertEqual(result["best_epoch_by_validation_f1"], 1)
            self.assertEqual(result["best_validation_macro_f1"], 0.8)
            self.assertEqual(evaluation_count[0], 3)  # 2 validation + exactly 1 test
            self.assertGreater(events.index("load_test"), events.index("fit_finished"))
            self.assertTrue((directory / "out/model.pt").is_file())
            self.assertEqual(result["implementation"], "corrected")
            self.assertEqual(result["contrastive_gradient"], "enabled")


if __name__ == "__main__":
    unittest.main()
