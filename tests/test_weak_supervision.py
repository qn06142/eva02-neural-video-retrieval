import argparse
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import SiglipConfig, SiglipModel

from training_utils import sigmoid_loss
from weak_supervision import (chunked_features, negative_pair_weights,
                              pair_diagnostics, FrozenTitleTeacher)


def tiny_model():
    return SiglipModel(SiglipConfig(
        text_config=dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=4),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=8)))


class WeakSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_joint_chunked_loss_and_all_parameter_gradients_match_full_batch(self):
        torch.manual_seed(0)
        original = tiny_model().train()
        chunked = copy.deepcopy(original)
        pixels = torch.randn(6, 3, 16, 16)
        text_inputs = dict(input_ids=torch.tensor([[2, 3, 1, 1], [4, 5, 1, 1], [6, 7, 1, 1]]),
                           attention_mask=torch.tensor([[1, 1, 1, 0]] * 3))

        def forward(model, size):
            image = chunked_features(model.get_image_features, dict(pixel_values=pixels), size)
            text = chunked_features(model.get_text_features, text_inputs, size)
            return sigmoid_loss(model, image.reshape(3, 2, -1), text, pooling="logmeanexp")

        full_loss, chunk_loss = forward(original, None), forward(chunked, 2)
        self.assertTrue(torch.allclose(full_loss, chunk_loss, atol=1e-6))
        full_loss.backward()
        chunk_loss.backward()
        for (name, a), (_, b) in zip(original.named_parameters(), chunked.named_parameters()):
            self.assertIsNotNone(a.grad, name)
            self.assertIsNotNone(b.grad, name)
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=3e-6, rtol=3e-4), name)

    def test_checkpoint_rng_replay_matches_same_chunking_with_dropout(self):
        module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.4), torch.nn.Linear(8, 4))
        other = copy.deepcopy(module)
        x = torch.randn(5, 4)
        torch.manual_seed(3)
        a = chunked_features(lambda values: module(values), dict(values=x), 2, recompute=False)
        a.sum().backward()
        next_a = torch.rand(3)
        torch.manual_seed(3)
        b = chunked_features(lambda values: other(values), dict(values=x), 2, recompute=True)
        b.sum().backward()
        next_b = torch.rand(3)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(next_a, next_b))
        for left, right in zip(module.parameters(), other.parameters()):
            self.assertTrue(torch.equal(left.grad, right.grad))

    def test_negative_weights_preserve_positives_and_known_negatives(self):
        # First two titles are identical in teacher space; third is orthogonal.
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
        positives = torch.eye(3, dtype=torch.bool)
        weights = negative_pair_weights(features, positives, .9, .25)
        self.assertEqual(weights[0, 1].item(), .25)
        self.assertTrue(torch.equal(weights.diag(), torch.ones(3)))
        self.assertEqual(weights[0, 2].item(), 1.)
        self.assertFalse(weights.requires_grad)
        stats = pair_diagnostics(weights, positives)
        self.assertEqual(stats["negative_pairs"], 6)
        self.assertEqual(stats["downweighted_negative_pairs"], 2)
        self.assertEqual(pair_diagnostics(None, positives)["negative_weight_sum"], 6)

    def test_pair_weighted_loss_preserves_positive_gradients(self):
        model = argparse.Namespace(logit_scale=torch.tensor(0.), logit_bias=torch.tensor(0.))
        image, text = torch.eye(2, requires_grad=True), torch.eye(2, requires_grad=True)
        base = sigmoid_loss(model, image, text)
        self.assertTrue(torch.equal(base, sigmoid_loss(model, image, text, pair_weights=torch.ones(2, 2))))
        reduced = sigmoid_loss(model, image, text, pair_weights=torch.zeros(2, 2))
        # Zero weights still leave the two diagonal positive losses at full strength.
        self.assertTrue(torch.allclose(reduced, torch.nn.functional.softplus(torch.tensor(-1.))))
        reduced.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        with self.assertRaises(ValueError):
            sigmoid_loss(model, image, text, pair_weights=torch.full((2, 2), float("nan")))

    def test_teacher_remains_frozen_and_resume_restores_original(self):
        model = tiny_model()
        ids = torch.tensor([[2, 3, 1, 1], [4, 5, 1, 1]])
        mask = torch.ones_like(ids)
        with tempfile.TemporaryDirectory() as tmp:
            teacher = FrozenTitleTeacher(model, tmp)
            expected = teacher(ids, mask).clone()
            with torch.no_grad():
                for param in model.text_model.parameters():
                    param.add_(1.)
            self.assertTrue(torch.equal(expected, teacher(ids, mask)))
            resumed = FrozenTitleTeacher(model, tmp, resume=True)
            self.assertEqual(resumed.sha256, teacher.sha256)
            self.assertTrue(torch.equal(expected, resumed(ids, mask)))
            self.assertTrue(all(not p.requires_grad for p in teacher.encoder.parameters()))
            with self.assertRaises(ValueError):
                FrozenTitleTeacher(model, tmp)

    def test_invalid_encoder_inputs_and_weight_settings(self):
        for inputs, size in ((dict(x=torch.empty(0, 4)), 2), (dict(x=torch.ones(2, 4)), 0)):
            with self.assertRaises(ValueError):
                chunked_features(lambda x: x, inputs, size)
        with self.assertRaises(ValueError):
            negative_pair_weights(torch.eye(2), torch.eye(2), 1., .25)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_amp_checkpointed_bag_backward(self):
        model = tiny_model().cuda().train()
        with torch.autocast("cuda", dtype=torch.float16):
            image = chunked_features(model.get_image_features,
                dict(pixel_values=torch.randn(6, 3, 16, 16, device="cuda")), 2)
            text = chunked_features(model.get_text_features,
                dict(input_ids=torch.tensor([[2, 3, 1, 1]]*3, device="cuda")), 2)
            loss = sigmoid_loss(model, image.reshape(3, 2, -1), text, pooling="logmeanexp")
        scaler = torch.amp.GradScaler("cuda", init_scale=16.)
        scaler.scale(loss).backward()
        self.assertTrue(torch.isfinite(model.vision_model.embeddings.patch_embedding.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
