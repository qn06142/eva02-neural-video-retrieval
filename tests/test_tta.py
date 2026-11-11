"""Offline TTA, pooling, data-schema and cache-identity regression tests."""
import argparse
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SiglipConfig, SiglipModel

from evaluate_tta import paired_interval
from retrieval_tta import (TTA_POLICIES, TEXT_TTA_MODES, cache_metadata, checkpoint_fingerprint,
    encode_tta_images, image_views, load_corpus, text_inputs, validate_cache,
    expand_text_query, encode_text_query)
from tests.test_training import FakeProcessor
from train_siglip import encode, load_pairs
from train_siglip_mil import encode_bags, mil_sigmoid_loss
from training_utils import pool_scores


class EvalProcessor(FakeProcessor):
    def __call__(self, text=None, images=None, **kwargs):
        result = {}
        if text is not None:
            result["input_ids"] = torch.tensor([[2, 3, 1, 1]] * len(text))
        if images is not None:
            result["pixel_values"] = super().__call__(["x"], images)["pixel_values"]
        return result


class TTATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        array = np.arange(6*8*3, dtype=np.uint8).reshape(6, 8, 3)
        self.image = Image.fromarray(array)

    def test_views_are_deterministic_and_flip_is_exact(self):
        for policy, count in zip(TTA_POLICIES, (1, 2, 5, 10)):
            a, b = image_views(self.image, policy, .75), image_views(self.image, policy, .75)
            self.assertEqual(len(a), count)
            self.assertTrue(all(x.tobytes() == y.tobytes() for x, y in zip(a, b)))
        flipped = np.asarray(image_views(self.image, "flip")[1])
        self.assertTrue(np.array_equal(flipped, np.asarray(self.image)[:, ::-1]))
        self.assertEqual(image_views(self.image, "five-crop", .75)[0].size, (6, 4))
        with self.assertRaises(ValueError):
            image_views(self.image, "none", float("nan"))

    def test_streamed_ensemble_matches_dense_and_bounds_batch(self):
        images = [self.image, self.image.rotate(180), self.image]
        sizes = []

        def pixels(batch):
            sizes.append(len(batch))
            return torch.stack([torch.tensor(np.array(im.resize((4, 4)))).flatten().float() for im in batch])

        def encoder(px):
            return torch.stack((px[:, 0]+1, px[:, 5]+3, px.mean(1)+2), dim=1)

        actual = encode_tta_images(images, pixels, encoder, "cpu", "ten-crop", .8, 7)
        self.assertLessEqual(max(sizes), 7)
        dense = encoder(pixels([view for image in images for view in image_views(image, "ten-crop", .8)]))
        expected = F.normalize(F.normalize(dense, dim=-1).reshape(3, 10, -1).mean(1), dim=-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        plain = encode_tta_images(images, pixels, encoder, "cpu", "none", .9, 2)
        self.assertTrue(torch.allclose(plain, F.normalize(encoder(pixels(images)), dim=-1)))

    def test_logmeanexp_bounds_invariance_and_gradient(self):
        scores = torch.tensor([[[.1, .3, .8]]], requires_grad=True)
        pooled = pool_scores(scores, "logmeanexp", .2)
        self.assertTrue((pooled <= scores.amax(-1)).all())
        self.assertTrue((pooled >= scores.mean(-1)).all())
        self.assertTrue(torch.allclose(pooled, pool_scores(scores.repeat(1, 1, 2), "logmeanexp", .2)))
        pooled.sum().backward()
        self.assertTrue((scores.grad > 0).all())
        with self.assertRaises(ValueError):
            pool_scores(scores, "logmeanexp", 0)
        model = argparse.Namespace(logit_scale=torch.tensor(1.), logit_bias=torch.tensor(-1.))
        img, txt = torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 8)
        mil_sigmoid_loss(model, img, txt, pooling="logmeanexp").backward()
        self.assertTrue(torch.isfinite(img.grad).all())

    def test_cache_rejects_wrong_model_policy_order_and_legacy(self):
        expected = cache_metadata("siglip", "abc", "flip", .9)
        blob = dict(metadata=expected, eids=["a", "b"], emb=torch.randn(2, 3, 8))
        validate_cache(blob, expected, ["a", "b"])
        for other in (dict(expected, checkpoint_sha256="changed"), dict(expected, tta="none")):
            with self.assertRaises(ValueError):
                validate_cache(blob, other, ["a", "b"])
        with self.assertRaises(ValueError):
            validate_cache(blob, expected, ["b", "a"])
        with self.assertRaises(ValueError):
            validate_cache(dict(emb=torch.randn(2, 8), eids=["a", "b"]), expected, ["a", "b"])
        mosaic = dict(expected, asset_type="mosaic", bag_size=9)
        with self.assertRaises(ValueError):
            validate_cache(dict(blob, metadata=mosaic), mosaic, ["a", "b"])

    def test_fingerprint_detects_changed_weights_at_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.safetensors"
            path.write_bytes(b"first")
            original = checkpoint_fingerprint(tmp)
            path.write_bytes(b"other")
            self.assertNotEqual(original, checkpoint_fingerprint(tmp))

    def test_live_and_legacy_crawler_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/"current.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE videos (eid TEXT,title TEXT,asset_state TEXT,asset_path TEXT)")
            conn.execute("INSERT INTO videos VALUES ('one','title','done',NULL)")
            conn.commit()
            conn.close()
            items, mode = load_corpus(db, tmp, tmp)
            self.assertEqual(mode, "mosaic")
            self.assertEqual(Path(items[0][0]), Path(tmp)/"one.jpg")
            self.assertEqual(load_pairs(str(db), tmp, None, 3), [])
            Image.new("RGB", (8, 8)).save(Path(tmp)/"one.jpg")
            self.assertEqual(len(load_pairs(str(db), tmp, None, 3)), 1)
            old = Path(tmp)/"old.db"
            conn = sqlite3.connect(old)
            conn.execute("CREATE TABLE videos (eid TEXT,title TEXT,thumb_state TEXT,thumb_path TEXT)")
            conn.execute("INSERT INTO videos VALUES ('one','title','done',NULL)")
            conn.commit()
            conn.close()
            self.assertEqual(load_corpus(old, tmp, tmp)[1], "thumbnail")

    def test_actual_siglip_tta_shapes_and_no_tta_equivalence(self):
        model = SiglipModel(SiglipConfig(
            text_config=dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=4),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8))).eval()
        proc = EvalProcessor()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"sheet.png")
            self.image.save(path)
            pairs = [(path, "sample one"), (path, "sample two")]
            images, texts = encode(model, proc, pairs, torch.device("cpu"), 2, 0)
            with torch.no_grad():
                reference = model(pixel_values=proc(images=[self.image]*2)["pixel_values"],
                                  **text_inputs(proc, [p[1] for p in pairs], "cpu"))
            self.assertTrue(torch.allclose(images, reference.image_embeds, atol=1e-6))
            self.assertTrue(torch.allclose(texts, reference.text_embeds, atol=1e-6))
            bags, _ = encode_bags(model, proc, pairs, torch.device("cpu"), 2, (2, 2), 2, 1,
                                  tta="ten-crop", view_batch_size=3)
            self.assertEqual(bags.shape, (2, 2, 16))
            self.assertTrue(torch.allclose(bags.norm(dim=-1), torch.ones(2, 2), atol=1e-6))


    def test_text_tta_expansion_and_modes(self):
        self.assertEqual(expand_text_query("lingerie", mode="none"), ["lingerie"])

        acc = expand_text_query("lingerie", mode="accurate")
        self.assertGreater(len(acc), 3)
        self.assertTrue(any("Explicit lingerie" in s for s in acc))

        vague = expand_text_query("lingerie", mode="vague")
        self.assertGreater(len(vague), 6)
        self.assertTrue(any("POV" in s for s in vague))
        self.assertTrue(any("1girl" in s for s in vague))

        with self.assertRaises(ValueError):
            expand_text_query("lingerie", mode="invalid_mode")

        # Test encode_text_query centroid vs multi
        fake_encoder = lambda texts: F.normalize(torch.randn(len(texts), 16), dim=-1)
        q_acc, mode_acc = encode_text_query(fake_encoder, "lingerie", mode="accurate")
        self.assertEqual(mode_acc, "accurate")
        self.assertEqual(q_acc.shape, (1, 16))
        self.assertTrue(torch.allclose(q_acc.norm(dim=-1), torch.ones(1), atol=1e-5))

        q_vague, mode_vague = encode_text_query(fake_encoder, "lingerie", mode="vague")
        self.assertEqual(mode_vague, "vague")
        self.assertEqual(q_vague.shape[0], len(vague))
        self.assertEqual(q_vague.shape[1], 16)

    def test_paired_bootstrap_preserves_pairing(self):
        self.assertEqual(paired_interval([1, 0, 1], [1, 0, 1], 100), dict(delta=0., low=0., high=0.))
        self.assertEqual(paired_interval([1, 1], [0, 0], 100), dict(delta=1., low=1., high=1.))


if __name__ == "__main__":
    unittest.main()
