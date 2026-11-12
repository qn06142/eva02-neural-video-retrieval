#!/usr/bin/env python3
"""Download the fine-tuned EVA-02 checkpoint from Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_REPO_ID = "wheatley1901/eva02-neural-video-retrieval"
HF_FILENAME = "eva02_curated_v2.pt"
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "scrape", "siglip_runs", "eva02_curated_v2", "best.pt")


def download_weights(
    output_path: str = DEFAULT_OUTPUT,
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
    force: bool = False,
) -> str:
    """Download checkpoint from Hugging Face Hub into `output_path`."""
    output_path = os.path.abspath(output_path)
    if not force and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[+] Weights already present at {output_path} ({size_mb:.2f} MB). Use --force to re-download.")
        return output_path

    dest_dir = os.path.dirname(output_path)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"[*] Downloading {repo_id}/{filename} from Hugging Face Hub...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        force_download=force,
    )

    downloaded_abs = os.path.abspath(downloaded_path)
    if downloaded_abs != output_path:
        # Move when hf_hub_download wrote a real file directly into dest_dir; otherwise copy safely.
        if os.path.dirname(downloaded_abs) == os.path.abspath(dest_dir) and not os.path.islink(downloaded_path):
            os.replace(downloaded_abs, output_path)
        else:
            tmp_path = f"{output_path}.tmp"
            try:
                shutil.copy2(os.path.realpath(downloaded_path), tmp_path)
                os.replace(tmp_path, output_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Saved weights to {output_path} ({size_mb:.2f} MB).")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EVA-02 fine-tuned weights from Hugging Face Hub.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Destination checkpoint path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--repo-id", default=HF_REPO_ID, help=f"Hugging Face model repo ID (default: {HF_REPO_ID})")
    parser.add_argument("--filename", default=HF_FILENAME, help=f"Checkpoint filename in repo (default: {HF_FILENAME})")
    parser.add_argument("--force", action="store_true", help="Re-download even if the destination file already exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    download_weights(
        output_path=args.output,
        repo_id=args.repo_id,
        filename=args.filename,
        force=args.force,
    )


if __name__ == "__main__":
    main()
