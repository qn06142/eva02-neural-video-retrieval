"""Filter and curate ~15,000 Danbooru2023 items specialized in Position/Pose and Clothing/Attire.
Source: scrape/danbooru_metadata/others/danbooru2023-tags.parquet
Manifest destination: scrape/danbooru2023_curated/manifest.sqlite3
"""

import os
import sqlite3
import json
import time
import pyarrow.parquet as pq

PARQUET_PATH = "scrape/danbooru_metadata/others/danbooru2023-tags.parquet"
DB_PATH = "scrape/danbooru2023_curated/manifest.sqlite3"
JSON_PATH = "scrape/danbooru2023_curated/curated_summary.json"

POSITION_TAGS = {
    # Core body orientations
    "kneeling", "sitting", "from_behind", "all_fours", "on_back", "on_stomach",
    "squatting", "standing", "lying", "spread_legs", "bent_over", "arched_back",
    # Specific pose/interaction dynamics
    "cowgirl_position", "doggystyle", "missionary", "spooning", "face-to-face",
    "straddling", "lifted_leg", "legs_up", "top-down_bottom-up", "side_view",
    "rear_view", "from_side", "from_above", "from_below", "fetal_position",
    "upright_straddle", "pinned", "against_wall", "standing_split", "leg_lock"
}

CLOTHING_TAGS = {
    # Lingerie & Underwear
    "lingerie", "underwear", "bra", "panties", "black_panties", "white_panties",
    "thighhighs", "black_thighhighs", "white_thighhighs", "stockings",
    "fishnet_stockings", "garter_straps", "babydoll",
    # Outfits & Uniforms
    "dress", "maid", "maid_uniform", "school_uniform", "serafuku", "sailor_collar",
    "bikini", "swimsuit", "one-piece_swimsuit", "competition_swimsuit",
    "bodysuit", "leotard", "bunny_suit", "nurse", "cheongsam", "china_dress",
    "kimono", "yukata", "crop_top", "skirt", "pleated_skirt", "micro_skirt",
    "shorts", "booty_shorts", "naked_apron", "naked_ribbon", "naked_overalls",
    "bridal_veil", "veil", "wedding_dress", "bare_shoulders", "collar"
}

EXCLUDE_TAGS = {
    "bad_anatomy", "bad_hands", "lowres", "comic", "manga", "greyscale",
    "monochrome", "traditional_media", "photo", "animated", "sample_watermark",
    "sketch", "rough_sketch"
}

def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS danbooru_curated (
            post_id INTEGER PRIMARY KEY,
            bucket INTEGER NOT NULL,
            rating TEXT NOT NULL,
            fav INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            aspect_ratio REAL NOT NULL,
            character TEXT,
            artist TEXT,
            position_tags TEXT NOT NULL,
            clothing_tags TEXT NOT NULL,
            all_tags TEXT NOT NULL,
            has_both INTEGER NOT NULL,
            is_downloaded INTEGER DEFAULT 0,
            image_path TEXT,
            file_bytes INTEGER DEFAULT 0
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bucket ON danbooru_curated(bucket)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_has_both ON danbooru_curated(has_both)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_downloaded ON danbooru_curated(is_downloaded)")
    conn.commit()
    return conn

def run_filter(target_count: int = 15000, min_fav: int = 20):
    print(f"Reading Parquet file: {PARQUET_PATH}")
    t0 = time.time()
    pf = pq.ParquetFile(PARQUET_PATH)
    num_rg = pf.num_row_groups
    print(f"Total row groups: {num_rg}")

    conn = init_db(DB_PATH)
    cur = conn.cursor()

    curated_both = []
    curated_pos_only = []
    curated_clothing_only = []

    total_scanned = 0
    for rg_idx in range(num_rg):
        rg_t0 = time.time()
        rg = pf.read_row_group(rg_idx, columns=['id', 'rating', 'fav', 'width', 'height', 'general', 'character', 'artist'])
        df = rg.to_pandas()
        n_rows = len(df)
        total_scanned += n_rows

        ids = df.index.to_numpy()
        favs = df['fav'].to_numpy()
        ratings = df['rating'].to_numpy()
        generals = df['general'].to_numpy()
        characters = df['character'].to_numpy()
        artists = df['artist'].to_numpy()
        widths = df['width'].to_numpy()
        heights = df['height'].to_numpy()

        for i in range(n_rows):
            fav = favs[i]
            if fav is None or fav < min_fav:
                continue

            r_arr = ratings[i]
            if r_arr is None or len(r_arr) == 0:
                continue
            rating = r_arr[0]
            if rating not in ('explicit', 'nsfw'):
                continue

            g_arr = generals[i]
            if g_arr is None or len(g_arr) == 0:
                continue

            gen_tags = set(t.replace(' ', '_') for t in g_arr)
            if gen_tags.intersection(EXCLUDE_TAGS):
                continue

            w = int(widths[i]) if widths[i] else 0
            h = int(heights[i]) if heights[i] else 0
            if w < 300 or h < 300:
                continue
            aspect = round(w / max(h, 1), 3)
            if aspect > 2.5 or aspect < 0.4:
                continue

            pos_matches = sorted(list(gen_tags.intersection(POSITION_TAGS)))
            clothing_matches = sorted(list(gen_tags.intersection(CLOTHING_TAGS)))

            if not pos_matches and not clothing_matches:
                continue

            post_id = int(ids[i])
            c_arr = characters[i]
            char_str = c_arr[0] if (c_arr is not None and len(c_arr) > 0) else ""
            a_arr = artists[i]
            art_str = a_arr[0] if (a_arr is not None and len(a_arr) > 0) else ""

            item = (
                post_id,
                int(post_id % 1000),
                rating,
                int(fav),
                w,
                h,
                aspect,
                char_str,
                art_str,
                json.dumps(pos_matches),
                json.dumps(clothing_matches),
                json.dumps(sorted(list(gen_tags))),
                1 if (pos_matches and clothing_matches) else 0
            )

            if pos_matches and clothing_matches:
                curated_both.append(item)
            elif pos_matches:
                curated_pos_only.append(item)
            else:
                curated_clothing_only.append(item)

        print(f"Row group {rg_idx+1}/{num_rg} scanned in {time.time()-rg_t0:.1f}s | "
              f"Both: {len(curated_both):,}, Pos: {len(curated_pos_only):,}, Clothing: {len(curated_clothing_only):,}")

    print(f"\nScan complete in {time.time()-t0:.1f}s across {total_scanned:,} total posts!")
    print(f"Candidates found: Dual-tagged={len(curated_both):,}, Position-only={len(curated_pos_only):,}, Clothing-only={len(curated_clothing_only):,}")

    # Sort each pool by community favorite count (quality descending)
    curated_both.sort(key=lambda x: x[3], reverse=True)
    curated_pos_only.sort(key=lambda x: x[3], reverse=True)
    curated_clothing_only.sort(key=lambda x: x[3], reverse=True)

    # Strategy: Select top 8,000 dual-tagged + 3,500 position-rich + 3,500 clothing-rich = 15,000 total
    target_both = min(len(curated_both), int(target_count * 0.55))
    target_pos = min(len(curated_pos_only), int((target_count - target_both) / 2))
    target_cloth = min(len(curated_clothing_only), target_count - target_both - target_pos)

    selected = curated_both[:target_both] + curated_pos_only[:target_pos] + curated_clothing_only[:target_cloth]
    print(f"Selected final set of {len(selected):,} items (Dual: {target_both:,}, Pos: {target_pos:,}, Clothing: {target_cloth:,})")

    # Insert into database
    cur.execute("DELETE FROM danbooru_curated")
    cur.executemany("""
        INSERT OR REPLACE INTO danbooru_curated 
        (post_id, bucket, rating, fav, width, height, aspect_ratio, character, artist, position_tags, clothing_tags, all_tags, has_both)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, selected)
    conn.commit()
    conn.close()

    summary = {
        "total_selected": len(selected),
        "dual_tagged": target_both,
        "position_tagged": target_pos,
        "clothing_tagged": target_cloth,
        "min_fav_threshold": min_fav,
        "sample_posts": [
            {
                "id": x[0],
                "bucket": x[1],
                "rating": x[2],
                "fav": x[3],
                "char": x[7],
                "pos": json.loads(x[9]),
                "clothing": json.loads(x[10])
            }
            for x in selected[:10]
        ]
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved manifest to {DB_PATH} and summary to {JSON_PATH}")

if __name__ == "__main__":
    run_filter(target_count=15000, min_fav=20)
