"""Unit and integration test verifying Colab G4 batch packaging, encoding, and merging.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_jpeg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scrape"))


class ColabMosaicDataset(Dataset):
    def __init__(self, item_list: list[dict], mosaics_dir: str):
        self.items = item_list
        self.mosaics_dir = mosaics_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.items[idx]
        img_path = os.path.join(self.mosaics_dir, entry["filename"])
        try:
            with open(img_path, "rb") as f:
                raw_bytes = bytearray(f.read())
            raw_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
            im = decode_jpeg(raw_tensor, mode=ImageReadMode.RGB)
            _, h, w = im.shape
            tw, th = w // 3, h // 3
            crops = torch.stack(
                [im[:, r * th : (r + 1) * th, c * tw : (c + 1) * tw] for r in range(3) for c in range(3)],
                dim=0,
            )
            resized = F.interpolate(
                crops.float(), size=(336, 336), mode="bilinear", align_corners=False
            ).to(torch.uint8)
        except Exception:
            resized = torch.zeros(9, 3, 336, 336, dtype=torch.uint8)
        return resized, idx


def collate_crops(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, list[int]]:
    tensors, indices = zip(*batch)
    return torch.cat(tensors, dim=0), list(indices)


class TestColabPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_colab_")
        self.mosaics_dir = os.path.join(self.test_dir, "mosaics")
        os.makedirs(self.mosaics_dir, exist_ok=True)

        # Generate 2 synthetic test mosaics (different dimensions)
        self.items = []
        # Item 1: 960x540
        img1 = (torch.rand(3, 540, 960) * 255).to(torch.uint8)
        from torchvision.io import write_jpeg
        p1 = os.path.join(self.mosaics_dir, "mosaic_01.jpg")
        write_jpeg(img1, p1)
        self.items.append({
            "eid": "colab_test_01",
            "title": "Colab Test Video 1",
            "filename": "mosaic_01.jpg",
            "asset_path": p1,
            "preview_url": "https://example.com/p1.mp4",
        })

        # Item 2: 1104x624
        img2 = (torch.rand(3, 624, 1104) * 255).to(torch.uint8)
        p2 = os.path.join(self.mosaics_dir, "mosaic_02.jpg")
        write_jpeg(img2, p2)
        self.items.append({
            "eid": "colab_test_02",
            "title": "Colab Test Video 2",
            "filename": "mosaic_02.jpg",
            "asset_path": p2,
            "preview_url": "https://example.com/p2.mp4",
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dataset_and_collate(self):
        ds = ColabMosaicDataset(self.items, self.mosaics_dir)
        self.assertEqual(len(ds), 2)

        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_crops)
        flat_batch, indices = next(iter(dl))

        self.assertEqual(flat_batch.shape, (18, 3, 336, 336))
        self.assertEqual(indices, [0, 1])
        self.assertEqual(flat_batch.dtype, torch.uint8)

    def test_mock_encoding_and_export(self):
        ds = ColabMosaicDataset(self.items, self.mosaics_dir)
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_crops)

        all_9 = []
        all_1 = []
        processed_indices = []

        for flat_uint8, batch_indices in dl:
            B = len(batch_indices)
            # Simulate embedding vector (B*9, 768)
            simulated_feats = torch.randn(B * 9, 768, dtype=torch.float32)
            simulated_feats = F.normalize(simulated_feats, dim=-1).half()
            tiles_9 = simulated_feats.view(B, 9, 768).cpu().numpy()
            tiles_1 = tiles_9[:, 4, :].copy()

            all_9.append(tiles_9)
            all_1.append(tiles_1)
            processed_indices.extend(batch_indices)

        final_9 = np.concatenate(all_9, axis=0)
        final_1 = np.concatenate(all_1, axis=0)

        self.assertEqual(final_9.shape, (2, 9, 768))
        self.assertEqual(final_1.shape, (2, 768))
        self.assertEqual(final_9.dtype, np.float16)

        out_9 = os.path.join(self.test_dir, "encoded_keyframes_9tiles.npy")
        out_1 = os.path.join(self.test_dir, "encoded_globals.npy")
        out_manifest = os.path.join(self.test_dir, "encoded_manifest.json")

        np.save(out_9, final_9)
        np.save(out_1, final_1)

        manifest_data = [self.items[i] for i in processed_indices]
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        self.assertTrue(os.path.exists(out_9))
        self.assertTrue(os.path.exists(out_1))
        self.assertTrue(os.path.exists(out_manifest))

        # Test merge logic
        test_meta_db = os.path.join(self.test_dir, "metadata.sqlite3")
        conn = sqlite3.connect(test_meta_db)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE video_metadata (
                idx INTEGER PRIMARY KEY,
                eid TEXT UNIQUE,
                title TEXT,
                clothing_tags TEXT,
                position_tags TEXT,
                asset_path TEXT,
                preview_url TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Run merge simulation
        emb_9 = np.load(out_9)
        emb_1 = np.load(out_1)
        with open(out_manifest, "r", encoding="utf-8") as f:
            m_items = json.load(f)

        self.assertEqual(emb_9.shape[0], len(m_items))

        conn = sqlite3.connect(test_meta_db)
        cur = conn.cursor()
        for i, item in enumerate(m_items):
            cur.execute("""
                INSERT INTO video_metadata (idx, eid, title, clothing_tags, position_tags, asset_path, preview_url)
                VALUES (?, ?, ?, '[]', '[]', ?, ?)
            """, (i, item["eid"], item["title"], item["asset_path"], item["preview_url"]))
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM video_metadata")
        count = cur.fetchone()[0]
        conn.close()

        self.assertEqual(count, 2)

    def test_magic_byte_tar_extraction_and_memmap(self):
        # Create an uncompressed tar archive intentionally named .tar.gz
        archive_path = os.path.join(self.test_dir, "misnamed_batch.tar.gz")
        src_manifest = os.path.join(self.test_dir, "manifest.json")
        with open(src_manifest, "w", encoding="utf-8") as f:
            json.dump(self.items, f)

        with tarfile.open(archive_path, "w") as tar:
            tar.add(src_manifest, arcname="manifest.json")
            for item in self.items:
                tar.add(item["asset_path"], arcname=f"mosaics/{item['filename']}")

        # Verify magic-byte check distinguishes uncompressed tar from gzip even if named .tar.gz
        with open(archive_path, "rb") as f:
            magic = f.read(2)
        is_gzip = magic == b"\x1f\x8b"
        self.assertFalse(is_gzip)

        extract_dir = os.path.join(self.test_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        mode = "r:gz" if is_gzip else "r:"
        with tarfile.open(archive_path, mode, bufsize=4 * 1024 * 1024) as tar:
            kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
            for member in tar:
                tar.extract(member, path=extract_dir, **kwargs)

        self.assertTrue(os.path.exists(os.path.join(extract_dir, "manifest.json")))
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "mosaics", "mosaic_01.jpg")))

        # Verify open_memmap writes standard np.load-compatible .npy arrays
        mm_9_path = os.path.join(self.test_dir, "mem_9.npy")
        mm_9 = np.lib.format.open_memmap(mm_9_path, mode="w+", dtype=np.float16, shape=(2, 9, 768))
        mm_9[0:2] = np.ones((2, 9, 768), dtype=np.float16)
        mm_9.flush()
        del mm_9

        reloaded = np.load(mm_9_path)
        self.assertEqual(reloaded.shape, (2, 9, 768))
        self.assertEqual(reloaded.dtype, np.float16)
        self.assertAlmostEqual(float(reloaded[1, 4, 0]), 1.0, places=3)

    def test_merge_colab_alignment_and_web_viewer_shards(self):
        import merge_colab_results as merge_mod
        import web_viewer as wv_mod

        emb_dir = os.path.join(self.test_dir, "emb")
        os.makedirs(emb_dir, exist_ok=True)
        meta_db = os.path.join(emb_dir, "metadata_147k.sqlite3")

        # 1. Create base 147k-style index with 2 rows (idx 0, 1)
        base_9 = np.zeros((2, 9, 768), dtype=np.float16)
        base_9[0, 2, 0] = 1.0  # idx 0 matches dim 0 on tile 2
        base_9[1, 7, 1] = 1.0  # idx 1 matches dim 1 on tile 7
        base_1 = base_9[:, 4, :].copy()

        base_9_path = os.path.join(emb_dir, "keyframes_9tiles_147k.npy")
        base_1_path = os.path.join(emb_dir, "globals_147k.npy")
        base_9.tofile(base_9_path)
        base_1.tofile(base_1_path)

        with open(os.path.join(emb_dir, "encoding_state.json"), "w", encoding="utf-8") as f:
            json.dump({"completed_items": 2, "total_items": 2, "dim": 768}, f)
        with open(os.path.join(emb_dir, "keyframes_9tiles_state.json"), "w", encoding="utf-8") as f:
            json.dump({"completed_items": 2, "total_items": 2, "tiles": 9, "dim": 768}, f)

        conn = sqlite3.connect(meta_db)
        conn.execute("""
            CREATE TABLE video_metadata (
                idx INTEGER PRIMARY KEY, eid TEXT UNIQUE, title TEXT,
                clothing_tags TEXT, position_tags TEXT, asset_path TEXT, preview_url TEXT
            )
        """)
        conn.execute("INSERT INTO video_metadata VALUES (0, 'base_0', 'Base 0', '[]', '[]', 'p0.jpg', '')")
        conn.execute("INSERT INTO video_metadata VALUES (1, 'base_1', 'Base 1', '[]', '[]', 'p1.jpg', '')")
        conn.commit()
        conn.close()

        # Simulate 1 uncommitted orphaned binary vector from an aborted earlier run
        np.zeros((1, 9, 768), dtype=np.float16).tofile(os.path.join(emb_dir, "live_keyframes_9tiles.bin"))
        np.zeros((1, 768), dtype=np.float16).tofile(os.path.join(emb_dir, "live_globals.bin"))

        # 2. Merge Colab batch containing 1 duplicate (base_0) + 2 new vectors
        colab_9 = np.zeros((3, 9, 768), dtype=np.float16)
        colab_9[1, 6, 2] = 1.0  # new_0 -> idx 2, dim 2 on tile 6
        colab_9[2, 4, 3] = 1.0  # new_1 -> idx 3, dim 3 on tile 4 (center)
        colab_1 = colab_9[:, 4, :].copy()

        npy_9_in = os.path.join(self.test_dir, "c9.npy")
        npy_1_in = os.path.join(self.test_dir, "c1.npy")
        man_in = os.path.join(self.test_dir, "c_man.json")
        np.save(npy_9_in, colab_9)
        np.save(npy_1_in, colab_1)
        with open(man_in, "w", encoding="utf-8") as f:
            json.dump([
                {"eid": "base_0", "title": "Dup Base 0", "asset_path": "dup.jpg", "preview_url": ""},
                {"eid": "new_0", "title": "New 0", "asset_path": "n0.jpg", "preview_url": "p0.mp4"},
                {"eid": "new_1", "title": "New 1", "asset_path": "n1.jpg", "preview_url": "p1.mp4"},
            ], f)

        res = merge_mod.merge_colab_batch(
            emb_dir=emb_dir,
            meta_db=meta_db,
            npy_9tiles=npy_9_in,
            manifest_path=man_in,
            npy_1tile=npy_1_in,
        )
        self.assertEqual(res["added"], 2)
        self.assertEqual(res["skipped"], 1)
        self.assertEqual(res["total"], 4)

        # Verify SQLite and live_*.bin match 1-to-1 with zero gaps
        conn = sqlite3.connect(meta_db)
        rows = conn.execute("SELECT idx, eid FROM video_metadata ORDER BY idx").fetchall()
        conn.close()
        self.assertEqual(rows, [(0, "base_0"), (1, "base_1"), (2, "new_0"), (3, "new_1")])

        live_9 = np.fromfile(os.path.join(emb_dir, "live_keyframes_9tiles.bin"), dtype=np.float16).reshape(-1, 9, 768)
        self.assertEqual(live_9.shape, (2, 9, 768))
        self.assertEqual(float(live_9[0, 6, 2]), 1.0)

        # 3. Verify web_viewer shard loading & 9-tile + 1-tile search across base + live shards
        segs_1, c1 = wv_mod._open_index_pair(
            base_1_path, os.path.join(emb_dir, "live_globals.bin"),
            os.path.join(emb_dir, "encoding_state.json"), (768,), 4,
        )
        segs_9, c9 = wv_mod._open_index_pair(
            base_9_path, os.path.join(emb_dir, "live_keyframes_9tiles.bin"),
            os.path.join(emb_dir, "keyframes_9tiles_state.json"), (9, 768), 4,
        )
        self.assertEqual(c1, 4)
        self.assertEqual(c9, 4)

        wv_mod.STATE["mmaps_1tile"] = segs_1
        wv_mod.STATE["total_1tile"] = c1
        wv_mod.STATE["mmaps_9tiles"] = segs_9
        wv_mod.STATE["total_9tiles"] = c9
        wv_mod.STATE["globals_gpu"] = wv_mod._copy_segments_to_gpu(segs_1, c1, (768,), "cpu")
        wv_mod.STATE["tiles_gpu_flat"] = wv_mod._copy_segments_to_gpu(segs_9, c9, (9, 768), "cpu").view(c9 * 9, 768)
        wv_mod.STATE["tiles_gpu_count"] = c9

        # Query matching idx 2 (live shard) on tile 6 (dim 2)
        q2 = torch.zeros(1, 768)
        q2[0, 2] = 1.0
        idxs, scores, tiles, total, _ = wv_mod._search_9tile(q2, 2, "cpu")
        self.assertEqual(total, 4)
        self.assertEqual(idxs[0], 2)
        self.assertEqual(tiles[0], 6)

        # Query matching idx 3 (live shard) center tile (dim 3)
        q3 = torch.zeros(1, 768)
        q3[0, 3] = 1.0
        idxs1, scores1, total1 = wv_mod._search_1tile(q3, 2, "cpu")
        self.assertEqual(total1, 4)
        self.assertEqual(idxs1[0], 3)


if __name__ == "__main__":
    unittest.main()
