"""Unit tests for the Pixiv SigLIP refinement pipeline."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from transformers import SiglipConfig, SiglipModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scrape"))

from frame_model import SiglipBagModel
from train_siglip_pixiv import (
    clean_tag, synthesize_video_title, format_tag_list, format_pixiv_caption,
    load_pixiv_records, load_xvideos_replay_pairs,
    PixivAdaptedDataset, Collator, configure_parameter_freezing,
)
from tests.test_training import FakeProcessor


class PixivPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_clean_tag_and_format_tags(self):
        self.assertEqual(clean_tag("school_uniform"), "school uniform")
        self.assertEqual(clean_tag("  long_hair&#39;s  "), "long hair's")

        chars = ["juliana_(pokemon)"]
        gens = ["1girl", "breasts", "blush"]
        formatted = format_tag_list(chars, gens, shuffle=False, dropout=0.0)
        self.assertEqual(formatted, "juliana (pokemon), 1girl, breasts, blush")

        # Test max_tags
        truncated = format_tag_list(chars, gens, shuffle=False, dropout=0.0, max_tags=2)
        self.assertEqual(truncated, "juliana (pokemon), 1girl")

        # Test dropout: dropout=1.0 drops general tags, leaves character tags
        dropped = format_tag_list(chars, gens, shuffle=False, dropout=1.0)
        self.assertEqual(dropped, "juliana (pokemon)")

    def test_synthesize_video_title_and_caption_styles(self):
        chars = ["juliana_(pokemon)"]
        gens = ["1girl", "large_breasts", "oral", "deepthroat", "creampie", "school_uniform", "pov"]

        title = synthesize_video_title(chars, gens)
        self.assertIn("POV", title)
        self.assertIn("Juliana (Pokemon)", title)
        self.assertIn("big tits", title)
        self.assertTrue("blowjob" in title or "deepthroat" in title)
        self.assertIn("creampie", title)

        # Test caption styles
        cap_title = format_pixiv_caption(chars, gens, style="video-title")
        self.assertNotIn("|", cap_title)

        cap_hybrid = format_pixiv_caption(chars, gens, style="hybrid")
        self.assertIn("|", cap_hybrid)

        cap_tags = format_pixiv_caption(chars, gens, style="tag-list")
        self.assertIn("juliana (pokemon)", cap_tags)

    def test_load_pixiv_records_from_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir = tmp_path / "images"
            images_dir.mkdir()

            db_path = tmp_path / "manifest.sqlite3"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE images (id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, metadata TEXT)")

            img_file1 = images_dir / "hash1.webp"
            Image.new("RGB", (32, 32), (255, 0, 0)).save(img_file1)

            img_file2 = images_dir / "hash2.webp"
            Image.new("RGB", (32, 32), (0, 255, 0)).save(img_file2)

            meta1 = {
                "path": str(img_file1),
                "tags_and_ratings": {
                    "character tags": ["char_a"],
                    "general tags": ["tag1", "tag2", "tag3"],
                }
            }
            meta2 = {
                "path": str(img_file2),
                "tags_and_ratings": {
                    "character tags": [],
                    "general tags": ["tag_x", "tag_y"],
                }
            }

            conn.execute("INSERT INTO images VALUES (?, ?, ?)", ("id1", "hash1", json.dumps(meta1)))
            conn.execute("INSERT INTO images VALUES (?, ?, ?)", ("id2", "hash2", json.dumps(meta2)))
            conn.commit()
            conn.close()

            # min_tags=3 should filter out id2
            records = load_pixiv_records(tmp, min_tags=3)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["id"], "id1")

            # min_tags=2 should include both
            records_all = load_pixiv_records(tmp, min_tags=2)
            self.assertEqual(len(records_all), 2)

    def test_pixiv_dataset_and_collator(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_file = Path(tmp) / "test.webp"
            Image.new("RGB", (64, 64), (100, 150, 200)).save(img_file)

            record = {
                "id": "1",
                "sha256": "h1",
                "path": str(img_file),
                "char_tags": ["hero"],
                "gen_tags": ["sword", "shield", "cape"],
                "canonical": "hero, sword, shield, cape",
                "is_xvideos": False,
            }

            processor = FakeProcessor()
            ds_val = PixivAdaptedDataset([record], processor, train=False)
            batch = [ds_val[0], ds_val[0]]
            collated = Collator(processor)(batch)
            self.assertIn("pixel_values", collated)
            self.assertIn("input_ids", collated)
            self.assertIn("attention_mask", collated)
            self.assertIn("positive_mask", collated)
            self.assertTrue(collated["positive_mask"][0, 1])

    def test_parameter_freezing_with_mixer(self):
        config = SiglipConfig(
            text_config=dict(vocab_size=16, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=4),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8))
        model = SiglipBagModel(config)
        model.enable_frame_mixer(hidden_dim=8, layers=2, heads=2)

        args = argparse.Namespace(
            freeze_backbone=False, freeze_vision=False,
            freeze_text=False, train_frame_mixer=False,
        )
        configure_parameter_freezing(model, args)
        self.assertFalse(model.frame_mixer.gate.requires_grad)
        self.assertTrue(model.vision_model.embeddings.patch_embedding.weight.requires_grad)
        self.assertTrue(model.text_model.embeddings.token_embedding.weight.requires_grad)

        args_frozen = argparse.Namespace(
            freeze_backbone=True, freeze_vision=False,
            freeze_text=False, train_frame_mixer=True,
        )
        configure_parameter_freezing(model, args_frozen)
        self.assertTrue(model.frame_mixer.gate.requires_grad)
        self.assertFalse(model.vision_model.embeddings.patch_embedding.weight.requires_grad)
        self.assertFalse(model.text_model.embeddings.token_embedding.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
