"""CPU architecture/mapping tests. CUDA renderer integration is separate."""
import unittest
import torch
from general_models import SceneRelighter, flat_to_map, map_to_flat


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_layer_grid_mapping(self):
        values=torch.arange(2*5*7*3,dtype=torch.float32).reshape(1,-1,3)
        grid=[2,5,7]
        mapped=flat_to_map(values,grid)
        self.assertEqual(tuple(mapped.shape),(1,6,5,7))
        torch.testing.assert_close(map_to_flat(mapped,grid),values)
        with self.assertRaises(ValueError): flat_to_map(values,[2,5,8])

    def test_both_architectures_gradient_and_restore(self):
        for arch in ["sharp","unet"]:
            model=SceneRelighter(arch,layers=2,feature_channels=10,width=8)
            attrs=torch.randn(1,17,19,21)
            feats=torch.randn(1,10,11,13) if arch=="sharp" else None
            output=model(attrs,feats,30,[2,23,25])
            self.assertEqual(tuple(output.shape),(1,1150,3))
            output.square().mean().backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0)
            copied=SceneRelighter(**model.config)
            copied.load_state_dict(model.state_dict())
            torch.testing.assert_close(output,copied(attrs,feats,30,[2,23,25]))
            # A new image/attributes tensor can be evaluated with the same weights.
            alternate=copied(attrs+torch.randn_like(attrs)*0.5,feats,30,[2,23,25])
            self.assertFalse(torch.allclose(output,alternate))

    def test_learning_angle_and_scene_condition(self):
        for arch in ["sharp","unet"]:
            torch.manual_seed(3)
            model=SceneRelighter(arch,layers=1,feature_channels=4,width=8)
            inputs=[torch.randn(1,10,16,16),torch.randn(1,10,16,16)]
            features=torch.randn(1,4,8,8) if arch=="sharp" else None
            targets=[(0,0.,0.2),(0,90.,0.7),(1,0.,0.6),(1,90.,0.3)]
            optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
            def objective():
                return sum((model(inputs[s],features,a,[1,16,16])-v).square().mean()
                           for s,a,v in targets)
            before=objective().item()
            for _ in range(70):
                optimizer.zero_grad()
                loss=objective(); loss.backward(); optimizer.step()
            self.assertLess(objective().item(),before*0.15,arch)


if __name__=="__main__": unittest.main()
