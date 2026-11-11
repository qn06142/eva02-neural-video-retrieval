#!/usr/bin/env python3
"""Filter XVideos dataset specifically for Position/Pose and Clothing/Attire retrieval.

Filters and annotates videos in thumbs.db where titles describe:
  1. Clothing Type / Attire / State of Dress (e.g. lingerie, stockings, skirt, bikini, maid, latex, nude)
  2. Position / Posture / Viewpoint (e.g. doggystyle, cowgirl, missionary, kneeling, from behind, pov)

Saves curated pairs into `videos_curated` table in thumbs.db with structured tags.
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_DIR, "scrape", "thumbs.db")
OUT_JSON = os.path.join(PROJECT_DIR, "scrape", "xvideos_position_clothing_pairs.json")

# 1. Clothing / Attire / State of dress patterns
CLOTHING_PATTERNS = [
    (r"\blingerie\b", "lingerie"),
    (r"\b(?:fishnets?|fishnet stockings?)\b", "fishnets"),
    (r"\b(?:stockings?|thigh[- ]?highs?|tights|pantyhose)\b", "stockings"),
    (r"\b(?:mini[- ]?skirts?)\b", "miniskirt"),
    (r"\b(?:skirts?)\b", "skirt"),
    (r"\b(?:dresses?|sundresses?|gowns?)\b", "dress"),
    (r"\b(?:bikinis?|micro[- ]?bikinis?)\b", "bikini"),
    (r"\b(?:swimsuits?|swimwear|monokinis?)\b", "swimsuit"),
    (r"\b(?:panties|panty|thongs?|g-strings?|undies|underwear)\b", "panties"),
    (r"\b(?:bras?|bralettes?|corsets?|bustiers?)\b", "bra"),
    (r"\b(?:maids?|french maid|maid uniform)\b", "maid uniform"),
    (r"\b(?:school[- ]?girls?|school uniform|sailor collar|serafuku)\b", "school uniform"),
    (r"\b(?:bunnys?|bunny suit)\b", "bunny suit"),
    (r"\b(?:nurses?|nurse uniform)\b", "nurse uniform"),
    (r"\b(?:cheerleaders?)\b", "cheerleader"),
    (r"\b(?:cosplays?|costumes?)\b", "cosplay"),
    (r"\b(?:bodysuits?|leotards?|catsuits?)\b", "bodysuit"),
    (r"\b(?:latex|rubber)\b", "latex"),
    (r"\b(?:leather)\b", "leather"),
    (r"\b(?:sheer|see[- ]?through|transparent)\b", "sheer"),
    (r"\b(?:laces?)\b", "lace"),
    (r"\b(?:yoga pants|leggings?)\b", "leggings"),
    (r"\b(?:shorts?|booty shorts?)\b", "shorts"),
    (r"\b(?:jeans?|denim)\b", "jeans"),
    (r"\b(?:robes?|kimonos?)\b", "robe"),
    (r"\b(?:aprons?|naked apron)\b", "apron"),
    (r"\b(?:garters?|garter belt|garter straps?)\b", "garter straps"),
    (r"\b(?:heels?|high heels?|boots?)\b", "heels"),
    (r"\b(?:nude|naked|bare|topless|bottomless|undressed|stripped|stripping|flashing)\b", "nude"),
]

# 2. Position / Posture / Viewpoint patterns
POSITION_PATTERNS = [
    (r"\b(?:prone[- ]?bone|pronebone|prone position|flat on (?:her )?stomach|face[- ]?down(?: on bed)?|laying on (?:her )?stomach|lying on (?:her )?stomach|on stomach)\b", "prone bone"),
    (r"\b(?:doggys?(?:style)?|doggy-style|from behind|from-behind|from the back|rear[- ]?entry|back[- ]?shots?|sex from behind)\b", "doggystyle"),
    (r"\b(?:bent[- ]?over|bend[- ]?over|ass[- ]?up|butt[- ]?up|arching back|arched back|on all fours|all[- ]?fours)\b", "bent over"),
    (r"\b(?:reverse[- ]?cowgirl|reverse[- ]?riding|reverse upright straddle)\b", "reverse cowgirl"),
    (r"\b(?:cowgirls?|riding (?:cock|dick|him)?|ridden|bouncing on|straddling|straddle|upright straddle|girl on top|woman on top)\b", "cowgirl"),
    (r"\b(?:missionarys?|face[- ]?to[- ]?face|knees to chest|knees[- ]to[- ]chest|holding legs|leg hold)\b", "missionary"),
    (r"\b(?:kneelings?|on knees|on her knees|on his knees)\b", "kneeling"),
    (r"\b(?:standings?|against (?:the )?wall|picked up|held up|standing sex|standing missionary)\b", "standing"),
    (r"\b(?:standing splits?|doing (?:the )?splits?|in (?:a )?splits?|flexible splits?)\b", "standing split"),
    (r"\b(?:sittings?|in chair|on couch|on desk|on table|lap dance|lapdance|sitting on lap|sitting on person)\b", "sitting"),
    (r"\b(?:squattings?|squat|squatting cowgirl)\b", "squatting"),
    (r"\b(?:layings?|lyings?|on bed|on floor|on back|spread eagle|legs in the air|fetal position|lying on person)\b", "lying down"),
    (r"\b(?:spread legs|legs up|legs wide|legs spread|open legs|wide spread legs)\b", "spread legs"),
    (r"\b(?:spoonings?|side[- ]?fuck|side by side)\b", "spooning"),
    (r"\b(?:lotus position|lotus sex|half lotus)\b", "lotus"),
    (r"\b(?:scissoring|tribadism|tribbing)\b", "scissoring"),
    (r"\b(?:mating press)\b", "mating press"),
    (r"\b(?:deepthroats?|deep[- ]?throat|throat fuck)\b", "deepthroat"),
    (r"\b(?:blowjobs?|suck(?:ing)? cock|giving head|oral|cunnilingus|eating out)\b", "oral"),
    (r"\b(?:titfucks?|tit-fucks?|paizuri|boob job)\b", "titfuck"),
    (r"\b(?:anals?|anal fuck|butt fuck|anal sex|ass fuck)\b", "anal"),
    (r"\b(?:pov|first[- ]?person|first person)\b", "POV"),
    (r"\b(?:close[- ]?up|close up)\b", "close up"),
]


def clean_title(raw: str) -> str:
    """Sanitize title: strip URLs, domain watermarks, and codec/volume junk."""
    if not raw:
        return ""
    t = html.unescape(raw)
    t = re.sub(r"https?://\S+|www\.\S+|bit\.ly/\S+|onl\.la/\S+", "", t)
    t = re.sub(r"\b[\w\-]+(?:\.com|\.net|\.org|\.tv|\.me|\.xyz|\.club)\b", "", t, flags=re.I)
    t = re.sub(r"\b(?:ModelMedia Asia|JulesJordan|Brazzers|BangBros|NaughtyAmerica|RealityKings|AdultTime|EvilAngel|Blacked|Tushy|Vixen|Mofos|TeamSkeet)\b\s*[-:|]*\s*", "", t, flags=re.I)
    t = re.sub(r"\[(?:4k|hd|1080p|60fps|full video|uncensored|exclusive|home video)\]", "", t, flags=re.I)
    t = re.sub(r"\((?:4k|hd|1080p|60fps|full video|uncensored|exclusive|home video|compilation)\)", "", t, flags=re.I)
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"[!?:;,\.\-_|\/\(\)\[\]]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -:|._")
    return t


def extract_clothing(text: str) -> list[str]:
    found = []
    for pat, label in CLOTHING_PATTERNS:
        if re.search(pat, text, re.I):
            if label not in found:
                found.append(label)
    return found


def extract_position(text: str) -> list[str]:
    found = []
    for pat, label in POSITION_PATTERNS:
        if re.search(pat, text, re.I):
            if label not in found:
                found.append(label)
    return found


def main() -> int:
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT eid, title, asset_path FROM videos WHERE asset_state='done'").fetchall()
    total = len(rows)
    print(f"Scanned {total:,} completed videos.")

    curated = []
    both_count = 0
    clothing_only = 0
    position_only = 0

    for eid, raw, path in rows:
        if not path or not os.path.exists(path):
            continue

        c_title = clean_title(raw)
        words = c_title.split()
        if len(c_title) < 10 or len(words) < 3:
            continue

        # Drop spam / verification
        if re.search(r"\b(?:verification video|verify my profile|verify id|check bio|whatsapp|telegram)\b", c_title, re.I):
            continue

        # Extract semantic attributes
        clothing = extract_clothing(c_title)
        position = extract_position(c_title)

        if not clothing and not position:
            continue

        has_both = bool(clothing and position)
        if has_both:
            both_count += 1
        elif clothing:
            clothing_only += 1
        else:
            position_only += 1

        curated.append({
            "eid": eid,
            "raw_title": raw,
            "clean_title": c_title,
            "clothing_tags": clothing,
            "position_tags": position,
            "has_both": int(has_both),
            "asset_path": path,
        })

    print(f"\n--- Curated Dataset Statistics ---")
    print(f"Total Filtered Samples: {len(curated):,} ({len(curated)/total*100:.1f}% of raw corpus)")
    print(f"  * Has BOTH Position & Clothing: {both_count:,}")
    print(f"  * Clothing-Specific Pairs:      {clothing_only + both_count:,}")
    print(f"  * Position-Specific Pairs:      {position_only + both_count:,}")

    # Store into SQLite table
    print("\nWriting to SQLite table `videos_curated`...")
    conn.execute("DROP TABLE IF EXISTS videos_curated")
    conn.execute("""
        CREATE TABLE videos_curated (
            eid TEXT PRIMARY KEY,
            clean_title TEXT,
            clothing_tags TEXT,
            position_tags TEXT,
            has_both INTEGER,
            asset_path TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_curated_both ON videos_curated (has_both)")

    insert_rows = [
        (c["eid"], c["clean_title"], json.dumps(c["clothing_tags"]),
         json.dumps(c["position_tags"]), c["has_both"], c["asset_path"])
        for c in curated
    ]
    conn.executemany("INSERT INTO videos_curated VALUES (?, ?, ?, ?, ?, ?)", insert_rows)
    conn.commit()
    conn.close()
    print("Database table `videos_curated` created and populated successfully.")

    # Write summary JSON for training pipeline inspection
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(curated[:5000], f, indent=2, ensure_ascii=False)
    print(f"Exported sample curated manifest to {OUT_JSON}")

    # Print 10 samples
    print("\n--- 10 Representative Curated Pairs ---")
    for i, c in enumerate(curated[:10], 1):
        cl = ", ".join(c["clothing_tags"]) if c["clothing_tags"] else "None"
        pos = ", ".join(c["position_tags"]) if c["position_tags"] else "None"
        print(f"#{i:02d} [{c['eid']}] Clothing: [{cl}] | Position: [{pos}]")
        print(f"    Clean Title: {c['clean_title']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
