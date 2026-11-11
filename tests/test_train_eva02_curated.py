import unittest
import random
import os
import torch
from scrape.train_eva02_curated import (
    format_xvideos_caption,
    format_danbooru_caption,
    load_curated_data,
    JointCuratedDataset,
    DEFAULT_XVIDEOS_DB,
    DEFAULT_DANBOORU_DB,
)


class TestTrainEva02Curated(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(42)

    def test_format_xvideos_caption_diversity(self):
        title = "Sexy amateur babe riding cock"
        pos = ["kneeling", "from_behind"]
        cloth = ["lingerie", "stockings"]
        
        captions = set()
        for i in range(20):
            c = format_xvideos_caption(title, cloth, pos, rng=random.Random(i), diversity=True)
            self.assertTrue(len(c) > 0)
            captions.add(c)
        # Should generate multiple diverse syntactic variants
        self.assertGreater(len(captions), 5)

    def test_format_danbooru_caption_diversity(self):
        char = "watson amelia"
        pos = ["kneeling", "from_behind"]
        cloth = ["stockings", "dress"]
        all_t = ["1girl", "blonde_hair", "blue_eyes", "solo", "breasts"]

        captions = set()
        for i in range(20):
            c = format_danbooru_caption(char, pos, cloth, all_t, rng=random.Random(i), diversity=True)
            self.assertTrue(len(c) > 0)
            captions.add(c)
        # Should generate multiple diverse syntactic variants
        self.assertGreater(len(captions), 5)

    def test_load_curated_data_splits(self):
        if not os.path.exists(DEFAULT_XVIDEOS_DB) or not os.path.exists(DEFAULT_DANBOORU_DB):
            self.skipTest("Databases not found")

        train_items, val_items = load_curated_data(
            xvideos_db=DEFAULT_XVIDEOS_DB,
            danbooru_db=DEFAULT_DANBOORU_DB,
            val_split=0.1,
            danbooru_ratio=0.5,
            seed=42
        )
        self.assertGreater(len(train_items), 0)
        self.assertGreater(len(val_items), 0)

        # Ensure no ID leakage between train and val
        train_ids = set(x["id"] for x in train_items)
        val_ids = set(x["id"] for x in val_items)
        leakage = train_ids.intersection(val_ids)
        self.assertEqual(len(leakage), 0, f"Found {len(leakage)} leaking IDs between train and val!")

    def test_dataset_item_retrieval(self):
        mock_items = [
            {
                "id": "xv_test1",
                "domain": "xvideos",
                "title": "Busty blonde in lingerie",
                "clothing_tags": ["lingerie"],
                "position_tags": ["kneeling"],
                "character": "",
                "all_tags": [],
                "image_path": "scrape/dataset_inspection_samples/xvideos/full_mosaics/xv_01_73663177_mosaic.jpg",
                "needs_crop": True
            },
            {
                "id": "dan_test1",
                "domain": "danbooru",
                "title": "",
                "clothing_tags": ["stockings", "dress"],
                "position_tags": ["from_behind"],
                "character": "watson amelia",
                "all_tags": ["1girl", "solo"],
                "image_path": "scrape/dataset_inspection_samples/danbooru/danbooru_01_4280954.jpg",
                "needs_crop": False
            }
        ]
        
        # Test loading
        dataset = JointCuratedDataset(mock_items, transform=None, diversity=True, seed=42)
        self.assertEqual(len(dataset), 2)
        
        t0, c0, d0, id0 = dataset[0]
        self.assertEqual(d0, "xvideos")
        self.assertEqual(id0, "xv_test1")
        self.assertTrue(len(c0) > 0)

        t1, c1, d1, id1 = dataset[1]
        self.assertEqual(d1, "danbooru")
        self.assertEqual(id1, "dan_test1")
        self.assertTrue(len(c1) > 0)


if __name__ == "__main__":
    unittest.main()
