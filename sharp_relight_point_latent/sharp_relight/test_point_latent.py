"""CPU gradient/learning tests; not a gsplat CUDA integration test."""
import unittest
import torch
from point_latent import PointLatentRelighter
from train_point_latent import sample_angle


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_chunk_gradients(self):
        model = PointLatentRelighter(17, 8, 16, 2)
        full = model(60, 100)
        full.square().mean().backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        chunk = model(60, 5)
        chunk.square().mean().backward()
        torch.testing.assert_close(full, chunk)
        for p, g in zip(model.parameters(), grads):
            torch.testing.assert_close(p.grad, g, atol=1e-7, rtol=1e-4)
        self.assertGreater(model.features.grad.abs().sum().item(), 0)

    def test_learning_from_fixed_compositor(self):
        model = PointLatentRelighter(12, 8, 24, 2)
        weights = torch.softmax(torch.randn(8, 12), dim=-1)
        targets = {0: torch.rand(8, 3) * 0.3 + 0.1,
                   90: torch.rand(8, 3) * 0.3 + 0.6}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial_features = model.features.detach().clone()
        def loss_fn():
            return sum((weights @ model(a, 100)[0] - t).square().mean()
                       for a, t in targets.items())
        initial = loss_fn().item()
        for _ in range(250):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()
        self.assertLess(loss_fn().item(), initial * 0.15)
        self.assertFalse(torch.equal(initial_features, model.features))
        copy = PointLatentRelighter(**model.config)
        copy.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy(45, 100), model(45, 100))

    def test_balanced_resumable_angles(self):
        angles = [0, 30, 60, 90, 120, 150, 180]
        sequence = [sample_angle(step, angles, 42) for step in range(1, 22)]
        for offset in [0, 7, 14]:
            self.assertEqual(sorted(sequence[offset:offset+7]), angles)
        self.assertEqual(sequence[10:], [sample_angle(s, angles, 42) for s in range(11, 22)])


if __name__ == "__main__":
    unittest.main()
