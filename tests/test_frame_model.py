import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scrape"))

import torch
import torch.nn.functional as F
from transformers import SiglipModel

from frame_model import ResidualFrameMixer, SiglipBagModel
from tests.test_weak_supervision import tiny_model
from tests.test_tta import EvalProcessor
from train_siglip_mil import encode_bags
from training_utils import sigmoid_loss


class FrameModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def model(self):
        model = SiglipBagModel(tiny_model().config)
        model.enable_frame_mixer(hidden_dim=8, layers=2, heads=2)
        return model

    def test_zero_gate_preserves_original_cosine_features(self):
        mixer = ResidualFrameMixer(16, 8, 2, 2)
        features = torch.randn(3, 9, 16)
        expected = F.normalize(features, dim=-1)
        self.assertTrue(torch.allclose(mixer(features), expected, atol=1e-6))
        text = torch.randn(3, 16)
        model = self.model()
        self.assertTrue(torch.allclose(sigmoid_loss(model, features, text),
                                        sigmoid_loss(model, mixer(features), text), atol=1e-6))

    def test_permutation_equivariance_and_real_frame_interaction(self):
        mixer = ResidualFrameMixer(16, 8, 2, 2).eval()
        with torch.no_grad():
            mixer.gate.fill_(.1)
        features = torch.randn(2, 9, 16)
        order = torch.tensor([8, 5, 0, 4, 1, 7, 6, 2, 3])
        output = mixer(features)
        self.assertTrue(torch.allclose(mixer(features[:, order]), output[:, order], atol=1e-6))
        changed = features.clone()
        changed[:, 1:] = torch.randn_like(changed[:, 1:])
        self.assertFalse(torch.allclose(mixer(changed)[:, 0], output[:, 0], atol=1e-6))
        self.assertEqual(mixer(features[:, :1]).shape, (2, 1, 16))

    def test_frozen_backbone_head_learns_and_gate_is_not_dead(self):
        model = self.model()
        model.freeze_backbone()
        model.train()
        self.assertFalse(model.text_model.training)
        self.assertFalse(model.vision_model.training)
        self.assertTrue(model.frame_mixer.training)
        initial = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        features, text = torch.randn(3, 4, 16), torch.randn(3, 16)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = sigmoid_loss(model, model.mix_frame_features(features), text)
            loss.backward()
            self.assertNotEqual(model.frame_mixer.gate.grad.item(), 0.)
            if step == 1:
                self.assertGreater(model.frame_mixer.down.weight.grad.abs().sum().item(), 0.)
            optimizer.step()
        for name, p in model.named_parameters():
            if name in initial:
                self.assertIsNone(p.grad, name)
                self.assertTrue(torch.equal(p, initial[name]), name)

    def test_architecture_and_weights_roundtrip(self):
        model = self.model().eval()
        with torch.no_grad():
            model.frame_mixer.gate.fill_(.07)
        features = torch.randn(2, 9, 16)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored = SiglipBagModel.from_pretrained(tmp, local_files_only=True).eval()
            self.assertEqual(restored.config.frame_mixer, model.config.frame_mixer)
            self.assertTrue(torch.allclose(restored.mix_frame_features(features),
                                           model.mix_frame_features(features), atol=1e-6))
            restored.freeze_backbone()
            self.assertTrue(all(name.startswith("frame_mixer.") for name, p in restored.named_parameters()
                                if p.requires_grad))

    def test_legacy_backbone_checkpoint_loads_without_head(self):
        legacy = tiny_model().eval()
        with tempfile.TemporaryDirectory() as tmp:
            legacy.save_pretrained(tmp)
            loaded = SiglipBagModel.from_pretrained(tmp, local_files_only=True).eval()
            self.assertIsNone(loaded.frame_mixer)
            value = torch.randn(2, 9, 16)
            self.assertTrue(torch.equal(value, loaded.mix_frame_features(value)))
            for name, param in legacy.state_dict().items():
                self.assertTrue(torch.equal(param, loaded.state_dict()[name]), name)

    def test_evaluation_applies_mixer_after_frame_encoding(self):
        from PIL import Image
        model = self.model().eval()
        proc = EvalProcessor()
        with torch.no_grad():
            model.frame_mixer.gate.fill_(.2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"sheet.png")
            Image.new("RGB", (32, 32), (50, 100, 150)).save(path)
            pairs = [(path, "one"), (path, "two")]
            mixed, _ = encode_bags(model, proc, pairs, torch.device("cpu"), 4, (2, 2), 2, 0)
            from train_siglip_mil import MosaicBagDataset
            tiles, _ = MosaicBagDataset(pairs, proc, bag_size=4, grid=(2, 2))[0]
            with torch.no_grad():
                raw = model.get_image_features(pixel_values=proc(images=tiles*2)["pixel_values"])
                expected = model.mix_frame_features(F.normalize(raw, dim=-1).reshape(2, 4, -1))
            self.assertTrue(torch.allclose(mixed, expected, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_mixer_cuda_bfloat_inference_and_amp_training(self):
        features = torch.randn(2, 9, 16, device="cuda")
        mixer = ResidualFrameMixer(16, 8, 2, 2).cuda().to(torch.bfloat16).eval()
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(mixer(features)).all())
        mixer = ResidualFrameMixer(16, 8, 2, 2).cuda().train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = mixer(features)
            loss = (output * torch.randn_like(output)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(mixer.gate.grad))


if __name__ == "__main__":
    unittest.main()
