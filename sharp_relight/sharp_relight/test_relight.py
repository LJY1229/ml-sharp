"""CPU checks; these do not replace a real gsplat CUDA integration test."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from relight_mlp import RelightMLP, normalize_positions, predict_colors
from train_relight import discover_sources, make_splits, target_path, check_target


class Tests(unittest.TestCase):
    def test_dataset_layout_and_no_scene_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for category in ["indoor", "outdoor"]:
                for index in range(1, 51):
                    path = root / category / f"{category}_{index:03d}" / "source.jpg"
                    path.parent.mkdir(parents=True)
                    path.touch()
            scenes = discover_sources(root)
            self.assertEqual(len(scenes), 100)
            split = make_splits(scenes)
            self.assertEqual([len(split[k]) for k in ["train", "val", "test"]], [80, 10, 10])
            self.assertEqual(len(set(sum(split.values(), []))), 100)
            self.assertEqual(split, make_splits(scenes))
            self.assertEqual(target_path(root, "outdoor_050", 30),
                             root / "angle_030" / "outdoor_050_angle_030.png")
            image = root / "test.png"
            Image.new("RGB", (200, 100)).save(image)
            check_target(image, (200, 400))
            with self.assertRaises(ValueError):
                check_target(image, (200, 200))

    def test_chunked_backward_matches_full(self):
        torch.manual_seed(2)
        model = RelightMLP(hidden=16, layers=2, frequencies=2)
        torch.nn.init.normal_(model.head.weight, std=0.02)
        xyz = torch.randn(1, 31, 3)
        rgb = torch.rand(1, 31, 3) * 0.8 + 0.1
        full = model(xyz, rgb, 45.)
        full.square().mean().backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        chunked = predict_colors(model, xyz, rgb, 45., chunk_size=7)
        chunked.square().mean().backward()
        torch.testing.assert_close(full, chunked)
        for parameter, expected in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-4)

    def test_identity_initialization_and_normalization(self):
        model = RelightMLP(hidden=16)
        rgb = torch.rand(1, 20, 3)
        xyz = torch.randn(1, 20, 3)
        torch.testing.assert_close(model(xyz, rgb, 0), rgb.clamp(1e-4, 1-1e-4))
        torch.testing.assert_close(normalize_positions(xyz), normalize_positions(xyz * 3 + 5),
                                   atol=1e-5, rtol=1e-4)
        self.assertTrue(torch.isfinite(normalize_positions(torch.zeros_like(xyz))).all())

    def test_angle_learning_through_fixed_differentiable_compositor(self):
        # A small fixed weighted compositor tests network gradient flow and angular
        # conditioning on CPU. It is not a substitute for gsplat rasterization.
        torch.manual_seed(4)
        torch.set_num_threads(1)
        model = RelightMLP(hidden=24, layers=2, frequencies=1)
        xyz = torch.randn(12, 3)
        rgb = torch.full((12, 3), 0.4)
        weights = torch.softmax(torch.randn(6, 12), dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        def objective():
            return sum((weights @ model(xyz, rgb, angle) - value).square().mean()
                       for angle, value in [(0., 0.2), (90., 0.7)])

        before = objective().item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            optimizer.step()
        self.assertLess(objective().item(), before * 0.02)
        self.assertFalse(rgb.requires_grad)
        self.assertFalse(xyz.requires_grad)


if __name__ == "__main__":
    unittest.main()
