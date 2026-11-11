"""Download a rating-filtered Pixiv subset using authenticated archive byte ranges.

No model imports, GPU use, training, or complete image-shard downloads.
Ratings/tags are WD-tagger predictions, not independently verified labels.
"""
import argparse
import concurrent.futures as futures
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile
import threading
import time

import requests
from huggingface_hub import get_token
from PIL import Image

REPO = 'Amber-River/Pixiv-2.6M'
REVISION = '29a513b76ada6ac474d1c1368f17fd5ec0ac45b7'
BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
# Minor-related tags are always excluded, in addition to the explicit rating filter.
DENY = ('loli', 'shota', 'child', 'toddler', 'infant', 'shotacon', 'lolicon', 'younger')
# Tags that hurt training: censorship, text overlays, watermarks, comics/multi-panel.
BAD_SUBSTR = ('censor', 'mosaic', 'text', 'watermark', 'signature', 'artist_name', 'speech_bubble',
              'bubble', 'comic', 'manga', 'translat', 'subtitle', 'dialogue', 'username', 'logo',
              'caption', 'onomatopoeia', 'sound_effect', 'web_address', 'twitter', 'patreon',
              'pixiv_id', 'multiple_views', 'split_screen', 'borders', 'letterboxed', 'pillarboxed',
              'bar_censor', 'light_censor', 'convenient_censoring', 'heart_censor', 'novelty_censor')
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.headers.update({'Authorization': 'Bearer ' + get_token(),
                                      'User-Agent': 'PixivGeneralSubset/1.0'})
    return LOCAL.session


def fetch(path, size_limit, expected_range=None):
    for attempt in range(4):
        headers = {}
        if expected_range:
            start, size, total = expected_range
            headers['Range'] = f'bytes={start}-{start + size - 1}'
        try:
            with session().get(BASE + path, headers=headers, timeout=(15, 60), stream=True) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(f'Dataset access denied: HTTP {r.status_code}')
                if r.status_code == 429 or r.status_code >= 500:
                    raise ConnectionError(f'Temporary HTTP {r.status_code}')
                r.raise_for_status()
                if expected_range:
                    target = f'bytes {start}-{start + size - 1}/{total}'
                    if r.status_code != 206 or r.headers.get('Content-Range') != target:
                        raise ValueError('Server did not honor the exact requested byte range')
                content = bytearray()
                for block in r.iter_content(128 * 1024):
                    content.extend(block)
                    if len(content) > size_limit:
                        raise ValueError('Response exceeded the expected size')
                if expected_range and len(content) != size:
                    raise ValueError('Truncated byte-range response')
                return bytes(content)
        except PermissionError:
            raise
        except (requests.RequestException, ConnectionError, ValueError) as exc:
            if attempt == 3:
                # Request exception messages may contain signed redirect URLs.
                raise RuntimeError(f'Download failed: {type(exc).__name__}') from None
            time.sleep(2 ** attempt)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def cached_index(cache, folder, shard):
    path = cache / f'{folder}-{shard}.json'
    if not path.exists():
        data = fetch(f'{folder}/{shard}.json', 2_000_000)
        parsed = json.loads(data)
        if not isinstance(parsed.get('files'), dict):
            raise ValueError('Unexpected archive index format')
        path.write_bytes(data)
    return json.loads(path.read_bytes())


def filter_reason(record):
    if record.get('rating class') != ['explicit']:
        return 'rating'
    tags = record.get('general tags', [])
    if not tags:
        return 'missing_tags'
    for tag in tags:
        padded = '_' + tag.lower() + '_'
        low = tag.lower()
        if any('_' + word + '_' in padded for word in DENY):
            return 'minor_tag_exclusion'
        if low != 'uncensored' and any(s in low for s in BAD_SUBSTR):
            return 'censor_text_exclusion'
    return None


def candidates(cache, shard, index, counters):
    path = cache / f'tags-{shard}.tar'
    if not path.exists():
        data = fetch(f'tags/{shard}.tar', index['filesize'])
        if len(data) != index['filesize'] or hashlib.sha256(data).hexdigest() != index['hash_lfs']:
            raise ValueError('Tag metadata archive hash mismatch')
        path.write_bytes(data)
    result = []
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.json'):
                continue
            if member.size > 200_000:
                raise ValueError('Unexpectedly large tag record')
            data = archive.extractfile(member).read()
            info = index['files'].get(member.name)
            if not info or hashlib.sha256(data).hexdigest() != info['sha256']:
                raise ValueError('Tag record hash mismatch')
            record = json.loads(data)
            reason = filter_reason(record)
            counters[reason or 'eligible'] = counters.get(reason or 'eligible', 0) + 1
            if not reason:
                result.append((PurePosixPath(member.name).stem, record, info['sha256']))
    # Stable choice within each shard, independent of tar member order.
    result.sort(key=lambda x: hashlib.sha256(('pixiv-explicit-v2:' + x[0]).encode()).hexdigest())
    return result


def download_image(output, shard, image_index, candidate):
    name, tags, tag_hash = candidate
    entry = image_index['files'].get(name + '.webp')
    if not entry:
        return name, None, 'No matching image index entry'
    try:
        data = fetch(f'webp/{shard}.tar', entry['size'],
                     (entry['offset'], entry['size'], image_index['filesize']))
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Image SHA-256 mismatch')
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            if im.format != 'WEBP' or min(width, height) < 224 or width * height > 40_000_000:
                raise ValueError('Unsupported image format or dimensions')
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        target = output / 'images' / (digest + '.webp')
        if not target.exists():
            temporary = target.with_suffix('.' + name + '.part')
            temporary.write_bytes(data)
            temporary.replace(target)
        metadata = {'image_id': name, 'pixiv_post_id': name.split('_')[0],
                    'pixiv_url': 'https://www.pixiv.net/artworks/' + name.split('_')[0],
                    'repository': REPO, 'revision': REVISION, 'shard': shard,
                    'image_member': name + '.webp', 'image_byte_range': entry,
                    'tag_sha256': tag_hash, 'sha256': digest, 'path': str(target.resolve()),
                    'width': width, 'height': height, 'bytes': len(data),
                    'tags_and_ratings': tags, 'label_source': 'WD-tagger-v2 (swin-v2) predictions',
                    'selection': 'explicit-only, minor-tag exclusion, v1',
                    'collected_at_unix': time.time()}
        return name, metadata, None
    except PermissionError:
        raise
    except (RuntimeError, ValueError, OSError, Image.DecompressionBombError) as exc:
        return name, None, str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', default=10_000, type=int)
    parser.add_argument('--workers', default=6, type=int)
    args = parser.parse_args()
    if not get_token():
        parser.error('Log in locally with hf auth login first')
    if args.count < 1 or not 1 <= args.workers <= 12:
        parser.error('count must be positive; workers must be between 1 and 12')
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 8 * 2**30:
        parser.error('At least 8 GiB free disk space required')
    cache = args.output / 'metadata_cache'
    cache.mkdir(exist_ok=True)
    (args.output / 'images').mkdir(exist_ok=True)
    config = {'repository': REPO, 'revision': REVISION, 'selection': 'explicit-clean-v2'}
    config_path = args.output / 'selection.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output directory contains a different subset configuration')
    atomic_json(config_path, config)
    db = sqlite3.connect(args.output / 'manifest.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, sha256 TEXT UNIQUE, metadata TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS failures (id TEXT PRIMARY KEY, reason TEXT)')
    db.commit()
    accepted = db.execute('SELECT count(*) FROM images').fetchone()[0]
    known = {row[0] for row in db.execute('SELECT id FROM images')}
    counters = {}
    started = time.time()
    progress = {'target': args.count, 'accepted': accepted, 'started_at_unix': started,
                'pid': os.getpid(), 'status': 'running', 'bytes': 0, 'failed': 0}

    def report():
        progress['accepted'] = accepted
        progress['updated_at_unix'] = time.time()
        progress['filter_counts'] = counters
        atomic_json(args.output / 'progress.json', progress)
        print(json.dumps({k: progress[k] for k in ('status', 'accepted', 'target', 'failed')})
              + f' shard={progress.get("shard", "-")}', flush=True)

    report()
    try:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for number in range(1100):
                if accepted >= args.count:
                    break
                if shutil.disk_usage(args.output).free < 4 * 2**30:
                    raise OSError('Stopped with less than 4 GiB disk space remaining')
                shard = f'data-{number:04d}'
                progress['shard'] = shard
                tag_index = cached_index(cache, 'tags', shard)
                selected = [x for x in candidates(cache, shard, tag_index, counters) if x[0] not in known]
                if not selected:
                    report()
                    continue
                image_index = cached_index(cache, 'webp', shard)
                # Bound the last shard to the number still needed.
                selected = selected[:args.count - accepted]
                pending = [pool.submit(download_image, args.output, shard, image_index, x) for x in selected]
                for task in futures.as_completed(pending):
                    name, metadata, error = task.result()
                    if error:
                        progress['failed'] += 1
                        db.execute('INSERT OR REPLACE INTO failures VALUES (?,?)', (name, error))
                    else:
                        result = db.execute('INSERT OR IGNORE INTO images VALUES (?,?,?)',
                                            (name, metadata['sha256'], json.dumps(metadata)))
                        if result.rowcount:
                            accepted += 1
                            progress['bytes'] += metadata['bytes']
                            known.add(name)
                            db.execute('DELETE FROM failures WHERE id=?', (name,))
                    db.commit()
                    if accepted % 50 == 0 or error:
                        report()
                report()
        if accepted < args.count:
            raise RuntimeError('Reached shard search limit before requested subset was complete')
        # A portable metadata export accompanies the durable SQLite manifest.
        temporary = args.output / 'metadata.jsonl.tmp'
        with temporary.open('w', encoding='utf-8') as stream:
            for (record,) in db.execute('SELECT metadata FROM images ORDER BY id'):
                stream.write(record + '\n')
        temporary.replace(args.output / 'metadata.jsonl')
        progress['status'] = 'complete'
        report()
    except BaseException as exc:
        progress['status'] = 'stopped'
        progress['error'] = type(exc).__name__ + ': ' + str(exc)
        report()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    main()


