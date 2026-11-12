"""Tests for Hugging Face checkpoint downloader, README instructions, and Colab fallback."""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scrape"))

import download_weights as dw_mod


class TestDownloadWeights(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_dl_weights_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_skips_when_existing_nonzero_without_force(self):
        dest = os.path.join(self.test_dir, "best.pt")
        with open(dest, "wb") as f:
            f.write(b"existing-checkpoint")

        with patch("download_weights.hf_hub_download") as mock_dl:
            out = dw_mod.download_weights(output_path=dest, force=False)

        self.assertEqual(out, os.path.abspath(dest))
        mock_dl.assert_not_called()
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"existing-checkpoint")

    def test_downloads_and_renames_from_local_dir(self):
        dest = os.path.join(self.test_dir, "nested", "best.pt")

        def fake_hf_dl(**kwargs):
            local_file = os.path.join(kwargs["local_dir"], kwargs["filename"])
            with open(local_file, "wb") as f:
                f.write(b"new-checkpoint-bytes")
            return local_file

        with patch("download_weights.hf_hub_download", side_effect=fake_hf_dl) as mock_dl:
            out = dw_mod.download_weights(output_path=dest, force=False)

        self.assertEqual(out, os.path.abspath(dest))
        mock_dl.assert_called_once_with(
            repo_id=dw_mod.HF_REPO_ID,
            filename=dw_mod.HF_FILENAME,
            local_dir=os.path.abspath(os.path.dirname(dest)),
            force_download=False,
        )
        self.assertTrue(os.path.exists(dest))
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(dest), dw_mod.HF_FILENAME)))
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"new-checkpoint-bytes")

    def test_copies_when_returned_path_is_outside_dest_dir_and_respects_force(self):
        cache_dir = os.path.join(self.test_dir, "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, "eva02_curated_v2.pt")
        with open(cached_file, "wb") as f:
            f.write(b"cached-weights-payload")

        dest = os.path.join(self.test_dir, "run", "best.pt")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"stale")

        with patch("download_weights.hf_hub_download", return_value=cached_file) as mock_dl:
            out = dw_mod.download_weights(output_path=dest, force=True)

        self.assertEqual(out, os.path.abspath(dest))
        mock_dl.assert_called_once()
        self.assertTrue(os.path.exists(cached_file))
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"cached-weights-payload")

    def test_cli_main_forwards_arguments(self):
        dest = os.path.join(self.test_dir, "cli_best.pt")
        with patch("download_weights.download_weights", return_value=dest) as mock_fn:
            dw_mod.main(["--output", dest, "--repo-id", "org/repo", "--filename", "ckpt.pt", "--force"])

        mock_fn.assert_called_once_with(
            output_path=dest,
            repo_id="org/repo",
            filename="ckpt.pt",
            force=True,
        )

    def test_readme_and_colab_include_hf_weights_download(self):
        readme_path = os.path.join(PROJECT_ROOT, "README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()

        self.assertIn("https://huggingface.co/wheatley1901/eva02-neural-video-retrieval", readme)
        self.assertIn("eva02_curated_v2.pt", readme)
        self.assertIn("python scrape/download_weights.py", readme)

        nb_path = os.path.join(PROJECT_ROOT, "colab_g4_encoder.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
        combined_code = "\n".join(code_cells)
        self.assertIn("wheatley1901/eva02-neural-video-retrieval", combined_code)
        self.assertIn("hf_hub_download", combined_code)

        # Ensure every pure Python cell (excluding cell 0 with shell ! magics) parses cleanly.
        for idx, src in enumerate(code_cells[1:], start=1):
            ast.parse(src, filename=f"colab_g4_encoder.ipynb#cell{idx}")


if __name__ == "__main__":
    unittest.main()
