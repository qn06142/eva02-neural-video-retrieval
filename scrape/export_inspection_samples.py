#!/usr/bin/env python3
"""Export a representative inspection subset from both XVideos and Danbooru2023 datasets.

Creates an inspection folder containing:
  - 50 XVideos curated samples (center action frame + full 3x3 mosaic) across positions & attire
  - 50 Danbooru2023 curated samples across postures, attire, and character compositions
  - An interactive visual gallery (`index.html`) displaying image, tags, character, and metadata
  - Structured manifest (`manifest.json`)
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import shutil
import sqlite3
import time
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, "scrape", "dataset_inspection_samples")
XVIDEOS_DB = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
DANBOORU_DB = os.path.join(PROJECT_DIR, "scrape", "danbooru2023_curated", "manifest.sqlite3")


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60].strip("_")


def main() -> int:
    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    xv_center_dir = os.path.join(OUT_DIR, "xvideos", "center_frames")
    xv_mosaic_dir = os.path.join(OUT_DIR, "xvideos", "full_mosaics")
    danbooru_out_dir = os.path.join(OUT_DIR, "danbooru")
    os.makedirs(xv_center_dir, exist_ok=True)
    os.makedirs(xv_mosaic_dir, exist_ok=True)
    os.makedirs(danbooru_out_dir, exist_ok=True)

    # Clean legacy pixiv directory if it exists
    old_pixiv_dir = os.path.join(OUT_DIR, "pixiv")
    if os.path.exists(old_pixiv_dir):
        shutil.rmtree(old_pixiv_dir, ignore_errors=True)

    manifest_entries = []

    # =========================================================================
    # 1. Sample 50 XVideos Curated Items
    # =========================================================================
    print("Selecting 50 representative XVideos curated samples...")
    conn = sqlite3.connect(XVIDEOS_DB)
    
    # 15 with both
    rows_both = conn.execute(
        "SELECT eid, clean_title, clothing_tags, position_tags, asset_path FROM videos_curated WHERE has_both=1 ORDER BY RANDOM() LIMIT 15"
    ).fetchall()

    # 15 clothing only
    rows_clothing = conn.execute(
        "SELECT eid, clean_title, clothing_tags, position_tags, asset_path FROM videos_curated WHERE has_both=0 AND clothing_tags != '[]' ORDER BY RANDOM() LIMIT 15"
    ).fetchall()

    # 20 position only
    rows_pos = conn.execute(
        "SELECT eid, clean_title, clothing_tags, position_tags, asset_path FROM videos_curated WHERE has_both=0 AND position_tags != '[]' ORDER BY RANDOM() LIMIT 20"
    ).fetchall()
    conn.close()

    xv_samples = rows_both + rows_clothing + rows_pos

    for idx, (eid, clean_title, c_json, p_json, asset_path) in enumerate(xv_samples, 1):
        c_tags = json.loads(c_json)
        p_tags = json.loads(p_json)
        
        slug = sanitize_filename(f"xv_{idx:02d}_{eid}")
        center_filename = f"{slug}_center.jpg"
        mosaic_filename = f"{slug}_mosaic.jpg"
        center_dest = os.path.join(xv_center_dir, center_filename)
        mosaic_dest = os.path.join(xv_mosaic_dir, mosaic_filename)

        if os.path.exists(asset_path):
            shutil.copy2(asset_path, mosaic_dest)
            try:
                with Image.open(asset_path) as im:
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    center = rgb.crop((w // 3, h // 3, 2 * w // 3, 2 * h // 3))
                    center.save(center_dest, quality=95)
            except Exception as e:
                print(f"Error cropping {asset_path}: {e}")

        manifest_entries.append({
            "id": f"xvideos_{eid}",
            "domain": "XVideos",
            "title": clean_title,
            "clothing_tags": c_tags,
            "position_tags": p_tags,
            "character_tags": [],
            "image_rel_path": f"xvideos/center_frames/{center_filename}",
            "mosaic_rel_path": f"xvideos/full_mosaics/{mosaic_filename}",
            "original_path": asset_path,
        })

    print(f"Exported {len(xv_samples)} XVideos inspection images.")

    # =========================================================================
    # 2. Sample 50 Danbooru2023 Curated Items
    # =========================================================================
    print("Selecting 50 representative Danbooru2023 curated samples...")
    conn = sqlite3.connect(DANBOORU_DB)
    
    # 20 with both
    rows_dan_both = conn.execute(
        "SELECT post_id, character, artist, position_tags, clothing_tags, image_path, fav FROM danbooru_curated WHERE is_downloaded=1 AND has_both=1 ORDER BY fav DESC LIMIT 20"
    ).fetchall()

    # 15 position only
    rows_dan_pos = conn.execute(
        "SELECT post_id, character, artist, position_tags, clothing_tags, image_path, fav FROM danbooru_curated WHERE is_downloaded=1 AND has_both=0 AND position_tags != '[]' ORDER BY fav DESC LIMIT 15"
    ).fetchall()

    # 15 clothing only
    rows_dan_cloth = conn.execute(
        "SELECT post_id, character, artist, position_tags, clothing_tags, image_path, fav FROM danbooru_curated WHERE is_downloaded=1 AND has_both=0 AND clothing_tags != '[]' ORDER BY fav DESC LIMIT 15"
    ).fetchall()
    conn.close()

    dan_samples = rows_dan_both + rows_dan_pos + rows_dan_cloth

    for idx, (post_id, char_name, artist_name, p_json, c_json, img_path, fav) in enumerate(dan_samples, 1):
        p_tags = [t.replace("_", " ") for t in json.loads(p_json)]
        c_tags = [t.replace("_", " ") for t in json.loads(c_json)]
        chars_clean = [char_name.replace("_", " ").title()] if char_name else []

        slug = sanitize_filename(f"danbooru_{idx:02d}_{post_id}")
        dest_filename = f"{slug}.jpg"
        dest_path = os.path.join(danbooru_out_dir, dest_filename)

        if img_path and os.path.exists(img_path):
            shutil.copy2(img_path, dest_path)

        caption_parts = []
        if chars_clean:
            caption_parts.append(f"Character: {chars_clean[0]}")
        if artist_name:
            caption_parts.append(f"Artist: {artist_name.replace('_', ' ')}")
        caption_parts.append(f"Favs: {fav}")

        title_desc = " | ".join(caption_parts)

        manifest_entries.append({
            "id": f"danbooru_{post_id}",
            "domain": "Danbooru2023",
            "title": title_desc,
            "clothing_tags": c_tags,
            "position_tags": p_tags,
            "character_tags": chars_clean,
            "image_rel_path": f"danbooru/{dest_filename}",
            "mosaic_rel_path": None,
            "original_path": img_path,
        })

    print(f"Exported {len(dan_samples)} Danbooru2023 inspection images.")

    # =========================================================================
    # 3. Write manifest.json
    # =========================================================================
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote manifest to {manifest_path}")

    # =========================================================================
    # 4. Generate Interactive Visual Gallery (index.html)
    # =========================================================================
    html_content = generate_html_gallery(manifest_entries)
    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated gallery HTML at {html_path}")

    return 0


def generate_html_gallery(entries: list[dict]) -> str:
    xv_count = sum(1 for e in entries if e["domain"] == "XVideos")
    dan_count = sum(1 for e in entries if e["domain"] == "Danbooru2023")

    cards_html = []
    for item in entries:
        dom = item["domain"]
        badge_cls = "badge-xv" if dom == "XVideos" else "badge-danbooru"
        
        pos_pills = "".join(f'<span class="pill pill-pos">{html.escape(p)}</span>' for p in item["position_tags"])
        cloth_pills = "".join(f'<span class="pill pill-cloth">{html.escape(c)}</span>' for c in item["clothing_tags"])
        char_pills = "".join(f'<span class="pill pill-char">{html.escape(ch)}</span>' for ch in item.get("character_tags", []))

        mosaic_btn = ""
        if item.get("mosaic_rel_path"):
            mosaic_btn = f'''
            <button class="btn-mosaic" onclick="openModal('{html.escape(item["mosaic_rel_path"])}')">
                View 3x3 Mosaic
            </button>
            '''

        card = f'''
        <div class="card" data-domain="{dom}" data-title="{html.escape(item['title'].lower())}" 
             data-pos="{html.escape(' '.join(item['position_tags']).lower())}" 
             data-cloth="{html.escape(' '.join(item['clothing_tags']).lower())}">
            <div class="img-box">
                <img src="{html.escape(item['image_rel_path'])}" alt="Thumbnail" loading="lazy" 
                     onclick="openModal('{html.escape(item['image_rel_path'])}')"/>
                <span class="badge {badge_cls}">{dom}</span>
            </div>
            <div class="card-content">
                <div class="card-title" title="{html.escape(item['title'])}">{html.escape(item['title'])}</div>
                <div class="tags-row">
                    {char_pills}
                    {pos_pills}
                    {cloth_pills}
                </div>
                {mosaic_btn}
            </div>
        </div>
        '''
        cards_html.append(card)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataset Inspection Gallery - XVideos (Curated) & Danbooru2023</title>
<style>
    :root {{
        --bg-main: #0d1117;
        --bg-card: #161b22;
        --border-color: #30363d;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-purple: #bc8cff;
        --accent-orange: #d29922;
        --accent-red: #f85149;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--bg-main);
        color: var(--text-primary);
        line-height: 1.5;
        padding: 24px;
    }}
    header {{
        max-width: 1400px;
        margin: 0 auto 24px auto;
        padding-bottom: 20px;
        border-bottom: 1px solid var(--border-color);
    }}
    h1 {{
        font-size: 26px;
        margin-bottom: 8px;
        color: #f0f6fc;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .subtitle {{
        color: var(--text-secondary);
        font-size: 14px;
        margin-bottom: 16px;
    }}
    .stats-bar {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 16px;
    }}
    .stat-badge {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 13px;
    }}
    .stat-badge b {{
        color: #fff;
    }}
    .controls {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
    }}
    .search-input {{
        flex: 1;
        min-width: 250px;
        background: #090d13;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 8px 14px;
        color: #fff;
        font-size: 14px;
    }}
    .filter-btn {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        padding: 8px 14px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        transition: all 0.2s;
    }}
    .filter-btn.active {{
        background: #238636;
        border-color: #2ea043;
        color: #fff;
    }}
    .grid {{
        max-width: 1400px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 20px;
    }}
    .card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .card:hover {{
        transform: translateY(-2px);
        border-color: var(--accent-blue);
    }}
    .img-box {{
        position: relative;
        width: 100%;
        height: 220px;
        background: #000;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}
    .img-box img {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }}
    .badge {{
        position: absolute;
        top: 8px;
        left: 8px;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
    }}
    .badge-xv {{ background: #da3633; color: #fff; }}
    .badge-danbooru {{ background: #1f6feb; color: #fff; }}
    .card-content {{
        padding: 12px;
        display: flex;
        flex-direction: column;
        flex: 1;
        gap: 8px;
    }}
    .card-title {{
        font-size: 13px;
        font-weight: 600;
        color: #f0f6fc;
        line-height: 1.4;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}
    .tags-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-top: auto;
    }}
    .pill {{
        font-size: 11px;
        padding: 2px 7px;
        border-radius: 12px;
        font-weight: 500;
    }}
    .pill-pos {{ background: rgba(210, 153, 34, 0.2); color: #e3b341; border: 1px solid rgba(210, 153, 34, 0.4); }}
    .pill-cloth {{ background: rgba(88, 166, 255, 0.2); color: #79c0ff; border: 1px solid rgba(88, 166, 255, 0.4); }}
    .pill-char {{ background: rgba(188, 140, 255, 0.2); color: #d2a8ff; border: 1px solid rgba(188, 140, 255, 0.4); }}
    .btn-mosaic {{
        background: transparent;
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 11px;
        cursor: pointer;
        text-align: center;
        margin-top: 4px;
        transition: all 0.2s;
    }}
    .btn-mosaic:hover {{
        border-color: var(--accent-blue);
        color: var(--accent-blue);
    }}
    /* Modal */
    .modal {{
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0; top: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.85);
        align-items: center;
        justify-content: center;
    }}
    .modal img {{
        max-width: 90vw;
        max-height: 90vh;
        border-radius: 6px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.8);
    }}
</style>
</head>
<body>

<header>
    <h1>Dataset Inspection Gallery</h1>
    <div class="subtitle">
        Visual inspection subset for evaluating pose & attire multi-modal training signal.
    </div>
    <div class="stats-bar">
        <div class="stat-badge">Total Inspected: <b>{len(entries)}</b></div>
        <div class="stat-badge">XVideos Curated: <b style="color: #f85149">{xv_count}</b></div>
        <div class="stat-badge">Danbooru2023 Curated: <b style="color: #58a6ff">{dan_count}</b></div>
        <div class="stat-badge">Yellow: <b>Pose/Position</b></div>
        <div class="stat-badge">Blue: <b>Clothing/Attire</b></div>
        <div class="stat-badge">Purple: <b>Character</b></div>
    </div>
    <div class="controls">
        <input type="text" id="searchInput" class="search-input" placeholder="Filter by keyword (e.g. kneeling, dress, stockings, maid, behind)..." oninput="applyFilters()" />
        <button class="filter-btn active" onclick="setFilter('all', this)">All ({len(entries)})</button>
        <button class="filter-btn" onclick="setFilter('XVideos', this)">XVideos ({xv_count})</button>
        <button class="filter-btn" onclick="setFilter('Danbooru2023', this)">Danbooru ({dan_count})</button>
        <button class="filter-btn" onclick="setFilter('both', this)">Dual Tagged</button>
    </div>
</header>

<div class="grid" id="galleryGrid">
    {''.join(cards_html)}
</div>

<div class="modal" id="modal" onclick="closeModal()">
    <img id="modalImg" src="" alt="Full view" />
</div>

<script>
    let currentFilter = 'all';

    function setFilter(filterType, btn) {{
        currentFilter = filterType;
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        applyFilters();
    }}

    function applyFilters() {{
        const search = document.getElementById('searchInput').value.toLowerCase().trim();
        const cards = document.querySelectorAll('.card');

        cards.forEach(card => {{
            const dom = card.getAttribute('data-domain');
            const title = card.getAttribute('data-title');
            const pos = card.getAttribute('data-pos');
            const cloth = card.getAttribute('data-cloth');
            const allText = title + ' ' + pos + ' ' + cloth;

            let matchesDomain = true;
            if (currentFilter === 'XVideos' && dom !== 'XVideos') matchesDomain = false;
            if (currentFilter === 'Danbooru2023' && dom !== 'Danbooru2023') matchesDomain = false;
            if (currentFilter === 'both' && (!pos || !cloth)) matchesDomain = false;

            let matchesSearch = true;
            if (search) {{
                matchesSearch = allText.includes(search);
            }}

            if (matchesDomain && matchesSearch) {{
                card.style.display = 'flex';
            }} else {{
                card.style.display = 'none';
            }}
        }});
    }}

    function openModal(src) {{
        const m = document.getElementById('modal');
        const img = document.getElementById('modalImg');
        img.src = src;
        m.style.display = 'flex';
    }}

    function closeModal() {{
        document.getElementById('modal').style.display = 'none';
    }}

    document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') closeModal();
    }});
</script>

</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
