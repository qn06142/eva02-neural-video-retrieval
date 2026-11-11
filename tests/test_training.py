"""Offline regression tests. Run: python -m unittest discover -s tests -t . -p test_training.py"""
import argparse
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import SiglipConfig, SiglipModel

from train_siglip import ThumbTitleDataset, Collator
from train_siglip_mil import MosaicBagDataset, BagCollator, mil_sigmoid_loss
from training_utils import (accumulation_weight, optimizer_step, positive_mask,
                            retrieval_recall, run_epochs, sigmoid_loss, split_samples)


class FakeProcessor:
    """Small picklable processor: no network, tokenizer assets or real images."""
    class Tokenizer:
        pad_token_id = 1
    tokenizer = Tokenizer()

    def __call__(self, text, images, **kwargs):
        return {"input_ids": torch.tensor([[2, 3, 1, 1]] * len(text)),
                "pixel_values": torch.stack([
                    torch.tensor(np.asarray(im.resize((16, 16))).copy()).permute(2, 0, 1).float()/255
                    for im in images])}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "model.pt")


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_split_groups_and_freezes(self):
        samples = [(f"{i}.jpg", f"title {i}") for i in range(30)]
        samples += [("duplicate.jpg", "TITLE 0"), ("duplicate.jpg", "other label")]
        with tempfile.TemporaryDirectory() as tmp:
            train, val = split_samples(samples, .25, 20, 42, tmp)
            self.assertEqual((train, val), split_samples(list(reversed(samples)), .25, 20, 42, tmp))
            group = [s for s in samples if s[0] in ("0.jpg", "duplicate.jpg")]
            self.assertTrue(all(s in train for s in group) or all(s in val for s in group))
            with self.assertRaises(ValueError):
                split_samples(samples + [("new.jpg", "new")], .25, 20, 42, tmp)

    def test_recall_matches_dense_and_handles_ties(self):
        torch.manual_seed(0)
        text, images = torch.randn(17, 8), torch.randn(17, 3, 8)
        scores = torch.einsum("id,jkd->ijk", torch.nn.functional.normalize(text, dim=-1),
                              torch.nn.functional.normalize(images, dim=-1)).amax(-1)
        order = scores.argsort(descending=True, dim=1)
        expected = {f"recall@{k}": (order[:, :k] == torch.arange(17)[:, None]).any(1).float().mean().item()
                    for k in (1, 5, 10)}
        actual = retrieval_recall(text, images, chunk_size=3)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])
        self.assertEqual(retrieval_recall(torch.zeros(4, 3), torch.zeros(4, 3))["recall@1"], .25)
        self.assertEqual(retrieval_recall(torch.eye(4), torch.eye(4))["recall@1"], 1)

    def test_loss_equivalence_duplicate_positives_and_amp(self):
        model = argparse.Namespace(logit_scale=torch.tensor(1., requires_grad=True),
                                   logit_bias=torch.tensor(-1., requires_grad=True))
        images = torch.randn(3, 2, 8, requires_grad=True)
        text = torch.randn(3, 8, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = mil_sigmoid_loss(model, images, text)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue(torch.isfinite(model.logit_scale.grad))
        self.assertTrue(positive_mask([" Same  title", "same title", "other"])[0, 1])
        normalized = torch.nn.functional.normalize(images.detach()[:, 0], dim=-1)
        txt = torch.nn.functional.normalize(text.detach(), dim=-1)
        logits = (txt @ normalized.T) * model.logit_scale.exp() + model.logit_bias
        targets = torch.eye(3) * 2 - 1
        reference = -torch.nn.functional.logsigmoid(targets * logits).sum(-1).mean()
        self.assertTrue(torch.allclose(sigmoid_loss(model, images[:, 0], text), reference))

    def test_partial_accumulation_sample_weights(self):
        # Three batches of 4,4,2 samples: first window weights 1/2,1/2;
        # final partial window must receive a full optimizer update.
        self.assertEqual([accumulation_weight(i, 10, 4, 2) for i in range(3)], [.5, .5, 1.])
        self.assertEqual([accumulation_weight(i, 6, 4, 2) for i in range(2)], [4/6, 2/6])

    def test_corrupt_images_fail_and_tiles_cover_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mosaic.png")
            Image.new("RGB", (11, 8)).save(path)
            ds = MosaicBagDataset([(path, "example")], None, bag_size=6, grid=(3, 2))
            tiles, title = ds[0]
            self.assertEqual(sum(t.width * t.height for t in tiles), 11*8)
            missing = str(Path(tmp) / "missing.jpg")
            with self.assertRaises(OSError):
                ThumbTitleDataset([(missing, "title")], None)[0]
            with self.assertRaises(OSError):
                MosaicBagDataset([(missing, "title")], None)[0]

    def test_actual_siglip_forward_backward_and_spawn_loader(self):
        config = SiglipConfig(
            text_config=dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=4),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8))
        model = SiglipModel(config)
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sheet.png")
            Image.new("RGB", (30, 30), (50, 100, 150)).save(path)
            for workers in (0, 1):
                ds = MosaicBagDataset([(path, "sample one"), (path, "sample two")], processor,
                                      bag_size=2, train=True)
                dl = DataLoader(ds, batch_size=2, num_workers=workers,
                                collate_fn=BagCollator(processor))
                enc, b, k = next(iter(dl))
                image = model.get_image_features(pixel_values=enc["pixel_values"])
                text = model.get_text_features(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                loss = mil_sigmoid_loss(model, image.reshape(b, k, -1), text, enc["positive_mask"])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
            batch = Collator(processor)([(Image.new("RGB", (16, 16)), "sample")] * 2)
            out = model(**{k: v for k, v in batch.items() if k != "positive_mask"})
            sigmoid_loss(model, out.image_embeds, out.text_embeds, batch["positive_mask"]).backward()

    def test_resume_reproduces_uninterrupted_cpu_run(self):
        def seed():
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)

        def args(out, resume=None):
            return argparse.Namespace(out=str(out), resume=resume, init_from=None, epochs=2,
                batch_size=2, grad_accum=2, weight_decay=.01, lr=.01, warmup_ratio=0.,
                clip=1., log_every=100, workers=0, device="cpu", skip_baseline=False,
                eval_tta="none", tta_crop_scale=.9, view_batch_size=64)

        def run(model, a, interrupt=False):
            dl = DataLoader(TensorDataset(torch.arange(7.).reshape(-1, 1)), batch_size=2,
                            shuffle=True, generator=torch.Generator().manual_seed(9))

            def loss(batch):
                noise = random.random() + np.random.rand() + torch.rand(1)
                return ((model.linear(batch[0]) - noise) ** 2).mean()

            evaluations = 0

            def evaluate():
                nonlocal evaluations
                evaluations += 1
                if interrupt and evaluations == 3:
                    raise RuntimeError("simulated interruption")
                return {"recall@1": .5}

            run_epochs(a, model, FakeProcessor(), dl, evaluate,
                       loss, torch.device("cpu"), lambda msg: None)

        with tempfile.TemporaryDirectory() as tmp:
            full, resumed = Path(tmp)/"full", Path(tmp)/"resumed"
            for path in (full, resumed):
                path.mkdir()
                (path/"split.json").write_text("same split")
            seed()
            model = TinyModel()
            run(model, args(full))
            seed()
            interrupted = TinyModel()
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run(interrupted, args(resumed), interrupt=True)
            # Check compatibility with checkpoints saved before optional TTA existed.
            state_path = resumed/"epoch-1"/"training_state.pt"
            old_state = torch.load(state_path, weights_only=False)
            for key in ("eval_tta", "tta_crop_scale", "view_batch_size"):
                old_state["args"].pop(key)
            torch.save(old_state, state_path)
            restored = TinyModel()
            restored.load_state_dict(torch.load(resumed/"epoch-1"/"model.pt", weights_only=True))
            run(restored, args(resumed, str(resumed/"epoch-1")))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            state = torch.load(resumed/"epoch-2"/"training_state.pt", weights_only=False)
            self.assertEqual(state["updates"], 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_fp16_loss_and_overflow_skip(self):
        parameters = argparse.Namespace(logit_scale=torch.tensor(2., device="cuda", requires_grad=True),
                                        logit_bias=torch.tensor(-2., device="cuda", requires_grad=True))
        images = torch.randn(3, 2, 8, device="cuda", dtype=torch.float16, requires_grad=True)
        text = torch.randn(3, 8, device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pair_loss = mil_sigmoid_loss(parameters, images, text)
        self.assertEqual(pair_loss.dtype, torch.float32)
        pair_loss.backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        model = TinyModel().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
        scaler = torch.amp.GradScaler("cuda", init_scale=128.)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.linear(torch.ones(2, 1, device="cuda")).sum()
        scaler.scale(loss).backward()
        self.assertTrue(optimizer_step(model, opt, sched, scaler, 1.))
        before = sched.last_epoch
        scaler.scale(model.linear(torch.ones(2, 1, device="cuda")).sum()).backward()
        for p in model.parameters():
            p.grad.fill_(float("inf"))
        self.assertFalse(optimizer_step(model, opt, sched, scaler, 1.))
        self.assertEqual(sched.last_epoch, before)


if __name__ == "__main__":
    unittest.main()
