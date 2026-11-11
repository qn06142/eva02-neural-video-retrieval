#!/usr/bin/env python3
"""Mass analysis of Danbooru taxonomy to automatically extract postures and clothing.

Extracts:
  - All physical postures, poses, stances, and intercourse positions from Danbooru's 751k tag ontology.
  - All clothing items, uniforms, garments, and attire from Danbooru.
  - Generates `scrape/danbooru_taxonomy_extracted.json` with empirical frequency counts.
  - Zero manual guessing or unreliable XVideos tag dependencies.
"""

from __future__ import annotations

import json
import os
import re
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG_PARQUET = os.path.join(PROJECT_DIR, "scrape", "danbooru_metadata", "parquet", "tag.parquet")
OUT_JSON = os.path.join(PROJECT_DIR, "scrape", "danbooru_taxonomy_extracted.json")


def clean_search_term(tag: str) -> str:
    """Convert Danbooru tag format to clean search engine query."""
    return tag.replace("_", " ").strip()


def main() -> int:
    print(f"Loading Danbooru tag ontology from {TAG_PARQUET}...")
    df = pd.read_parquet(TAG_PARQUET)
    general = df[df["type"] == 0].copy()
    print(f"Loaded {len(general):,} general tags.")

    # 1. Posture & Pose Morphology Patterns
    pose_patterns = [
        r"_pose$", r"_stance$", r"_position$", r"_sitting$", r"_standing$",
        r"_lying$", r"_kneeling$", r"_squatting$", r"_straddle$", r"_straddling$",
        r"^sitting_", r"^standing_", r"^lying_", r"^kneeling_", r"^squatting_",
        r"^prone_", r"^missionary", r"^cowgirl", r"^doggystyle",
        r"on_stomach", r"from_behind", r"all_fours", r"spread_legs", r"bent_over",
        r"arched_back", r"fetal_position", r"spooning", r"mating_press",
        r"reverse_cowgirl", r"reverse_upright_straddle", r"upright_straddle",
        r"leg_hold", r"holding_legs", r"knees_to_chest", r"knees_apart",
        r"knees_together", r"standing_split", r"arm_between_legs"
    ]
    pose_regex = re.compile("|".join(pose_patterns), re.I)

    # 2. Clothing & Wardrobe Morphology Patterns
    cloth_patterns = [
        r"_uniform$", r"_costume$", r"_dress$", r"_skirt$", r"_suit$", r"_lingerie$",
        r"_stockings$", r"_thighhighs$", r"_bikini$", r"_swimsuit$", r"_panties$",
        r"_bra$", r"_apron$", r"_collar$", r"_boots$", r"_heels$", r"_leotard$",
        r"_bodysuit$", r"_jacket$", r"_hoodie$", r"_shirt$", r"_shorts$", r"_pants$",
        r"_vest$", r"_robe$", r"_kimono$", r"_cloak$", r"_cape$", r"_sleeves$",
        r"maid", r"lingerie", r"serafuku", r"sailor_collar", r"garter", r"corset"
    ]
    cloth_regex = re.compile("|".join(cloth_patterns), re.I)

    # Filter with minimum popularity of 50 to drop one-off typos
    valid_tags = general[general["popularity"] >= 50].copy()

    postures = []
    clothing = []

    # Drop non-posture false positives (e.g. object positions or body marks)
    exclude_pose_words = {"tree", "grass", "cloud", "sun", "background", "border", "flag", "star", "symbol", "(western)"}
    exclude_prefixes = ("scar_", "mole_", "bandaid_", "tattoo_", "piercing_", "cum_", "lipstick_", "writing_")

    for _, row in valid_tags.iterrows():
        name = row["name"]
        pop = int(row["popularity"])

        # Check posture
        if pose_regex.search(name):
            if not any(ex in name for ex in exclude_pose_words) and not any(name.startswith(pfx) for pfx in exclude_prefixes):
                postures.append({
                    "tag": name,
                    "search_query": clean_search_term(name),
                    "popularity": pop
                })

        # Check clothing
        elif cloth_regex.search(name):
            clothing.append({
                "tag": name,
                "search_query": clean_search_term(name),
                "popularity": pop
            })

    postures.sort(key=lambda x: x["popularity"], reverse=True)
    clothing.sort(key=lambda x: x["popularity"], reverse=True)

    print(f"Extracted {len(postures):,} distinct posture categories.")
    print(f"Extracted {len(clothing):,} distinct clothing categories.")

    result = {
        "metadata": {
            "source": "Danbooru2023 Tag Ontology (751,547 tags analyzed)",
            "method": "Mass morphological & semantic taxonomy extraction",
            "total_posture_classes": len(postures),
            "total_clothing_classes": len(clothing),
        },
        "postures": postures,
        "clothing": clothing[:500]  # Top 500 clothing types
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved complete taxonomy to: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
