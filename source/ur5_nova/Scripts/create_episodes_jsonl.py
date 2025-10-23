import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import json
import re
from typing import Optional
import pathlib

DEFAULT_TASK_TEXT = "Push the T-shaped block onto the T-shaped target."

def _detect_chunk_dir(dataset_root: str) -> tuple[str, str]:
    """Return (chunk_dir_abs_path, chunk_dir_name).

    Supports both chunk_000 and chunk-000 naming. Raises if not found.
    """
    candidates = ["chunk_000", "chunk-000"]
    data_dir = os.path.join(dataset_root, "data")
    for c in candidates:
        p = os.path.join(data_dir, c)
        if os.path.isdir(p):
            return p, c
    raise FileNotFoundError(f"No chunk directory found under {data_dir}. Tried: {candidates}")

def _write_or_update_info_json(dataset_root: str, total_episodes: int, total_frames: int):
    """Create or update meta/info.json with aggregate counters.

    Keys managed:
      - total_episodes
      - total_frames
      - total_videos (assumed == total_episodes; per logical episode, not per camera)
      - total_chunks (directories under data/ named chunk_*/chunk-*)
    Other existing keys are preserved.
    """
    meta_dir = os.path.join(dataset_root, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    info_path = os.path.join(meta_dir, 'info.json')
    # Count chunks
    data_dir = os.path.join(dataset_root, 'data')
    total_chunks = 0
    if os.path.isdir(data_dir):
        for entry in os.listdir(data_dir):
            if entry.startswith('chunk') and os.path.isdir(os.path.join(data_dir, entry)):
                total_chunks += 1
    info = {}
    if os.path.isfile(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f) or {}
        except Exception:
            info = {}
    info.update({
        'total_episodes': int(total_episodes),
        'total_frames': int(total_frames),
        'total_videos': int(total_episodes),  # one logical video set per episode
        'total_chunks': int(total_chunks),
    })
    try:
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"[episodes_jsonl] Updated info.json -> {info_path}")
    except Exception as e:
        print(f"[episodes_jsonl] Failed to write info.json: {e}")

def generate_episodes_jsonl(dataset_root: str, task_text: Optional[str] = None) -> str:
    """Generate episodes.jsonl inside <dataset_root>/meta.

    Parameters
    ----------
    dataset_root : str
        Root of dataset (contains data/, videos/, etc.)
    task_text : Optional[str]
        Override default task description.

    Returns
    -------
    str : Path to generated episodes.jsonl
    """
    chunk_path, chunk_name = _detect_chunk_dir(dataset_root)
    # Collect parquet files (episodes)
    onlyfiles = [f for f in listdir(chunk_path) if isfile(join(chunk_path, f)) and f.endswith('.parquet')]
    onlyfiles.sort()
    if not onlyfiles:
        raise FileNotFoundError(f"No parquet episode files found in {chunk_path}")

    jsonl_data = []
    total_length = 0
    task_text = task_text or DEFAULT_TASK_TEXT

    for i, file in enumerate(onlyfiles):
        try:
            df = pd.read_parquet(os.path.join(chunk_path, file))
        except Exception as e:
            print(f"[episodes_jsonl] Skipping {file}: failed to read parquet ({e})")
            continue
        if 'index' not in df.columns:
            print(f"[episodes_jsonl] Skipping {file}: missing 'index' column")
            continue
        # Derive episode index: attempt to parse from filename first
        episode_idx = i
        episode_match = re.search(r'episode[_-]?(\d+)', file, re.IGNORECASE)
        if episode_match:
            try:
                episode_idx = int(episode_match.group(1))
            except ValueError:
                pass
        length = len(df)
        total_length += length
        episode_dic = {
            'episode_index': episode_idx,
            'tasks': [task_text],
            'length': length,
        }
        jsonl_data.append(episode_dic)
        print(f"[episodes_jsonl] {file}: episode_index={episode_idx} length={length}")

    meta_dir = os.path.join(dataset_root, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    out_path = os.path.join(meta_dir, 'episodes.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry))
            f.write('\n')
    print(f"[episodes_jsonl] Wrote {len(jsonl_data)} episodes (total length {total_length}) to {out_path}")
    # Update info.json with aggregate stats
    try:
        _write_or_update_info_json(dataset_root, len(jsonl_data), total_length)
    except Exception as e:
        print(f"[episodes_jsonl] info.json update failed: {e}")
    return out_path

if __name__ == '__main__':
    dataset_root = os.environ.get('DATASET_ROOT', os.path.join(os.environ['HOME'], 'training_data/lerobot/my_pusht'))
    try:
        generate_episodes_jsonl(dataset_root)
    except Exception as e:
        print(f"[episodes_jsonl] Failed: {e}")
