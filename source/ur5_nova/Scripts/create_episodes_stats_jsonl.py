import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import json
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import cv2
import shutil
import re
from typing import Optional, Tuple
## (duplicate json import removed)

def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))

def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array

def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": np.array([len(array)]).tolist(),
    }


def get_feature_stats_img(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }

def _detect_chunk_dir(dataset_root: str) -> Tuple[str, str]:
    candidates = ["chunk_000", "chunk-000"]
    data_dir = os.path.join(dataset_root, "data")
    for c in candidates:
        p = os.path.join(data_dir, c)
        if os.path.isdir(p):
            return p, c
    raise FileNotFoundError(f"No chunk directory found under {data_dir}. Tried: {candidates}")

def _detect_video_chunk_dir(dataset_root: str) -> Tuple[str, str]:
    candidates = ["chunk_000", "chunk-000"]
    base = os.path.join(dataset_root, 'videos')
    for c in candidates:
        # Accept multiple possible observation image folder conventions
        for leaf in ['observation.image', 'observation.image/', 'observation.images.top', 'observation.images.state', 'observation.images.wrist']:
            p = os.path.join(base, c, leaf)
            if os.path.isdir(p):
                return os.path.join(base, c), c
    # Fallback: just videos root
    return base, ''

def _find_any_episode_video(video_chunk_root: str, episode_basename: str) -> Optional[str]:
    """Try to locate a representative video file for an episode.

    Searches common subfolders (top, wrist, state, observation.image) and returns first existing path.
    """
    candidates_dirs = [
        'observation.image',
        'observation.images.top',
        'observation.images.state',
        'observation.images.wrist',
        'top', 'wrist', 'state'
    ]
    for d in candidates_dirs:
        pdir = os.path.join(video_chunk_root, d)
        if not os.path.isdir(pdir):
            continue
        fname = f"{episode_basename}.mp4"
        fpath = os.path.join(pdir, fname)
        if os.path.isfile(fpath):
            return fpath
    return None

def _extract_frames_to_temp(video_path: str, temp_dir: str) -> list[str]:
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    img_paths: list[str] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(temp_dir, f"img{frame_idx}.png")
        cv2.imwrite(out_path, frame)
        img_paths.append(out_path)
        frame_idx += 1
    cap.release()
    return img_paths

def generate_episodes_stats_jsonl(dataset_root: str) -> str:
    """Generate episodes_stats.jsonl in <dataset_root>/meta.

    Reads parquet episode files and (optionally) associated video frames to compute statistics.
    Falls back gracefully if video is missing (skips observation.image stats for that episode).
    """
    chunk_path, chunk_name = _detect_chunk_dir(dataset_root)
    video_chunk_root, _ = _detect_video_chunk_dir(dataset_root)
    parquet_files = [f for f in listdir(chunk_path) if isfile(join(chunk_path, f)) and f.endswith('.parquet')]
    parquet_files.sort()
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episode files found in {chunk_path}")
    jsonl_data = []
    total_frames_accum = 0
    for i, file in enumerate(parquet_files):
        episode_dic: dict = {}
        parquet_path = os.path.join(chunk_path, file)
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[episodes_stats] Skipping {file}: read error {e}")
            continue
        if 'index' not in df.columns:
            print(f"[episodes_stats] Skipping {file}: missing 'index' column")
            continue
        episode_idx = i
        episode_match = re.search(r'episode[_-]?(\d+)', file, re.IGNORECASE)
        if episode_match:
            try:
                episode_idx = int(episode_match.group(1))
            except ValueError:
                pass
        print(f"[episodes_stats] {file}: episode_index={episode_idx} frames={len(df)}")
        episode_dic['episode_index'] = episode_idx
        episode_dic['stats'] = {}
        total_frames_accum += len(df)
        # Observation image stats (optional if video present)
        episode_basename = file.rsplit('.', 1)[0]  # episode_00001
        video_path = _find_any_episode_video(video_chunk_root, episode_basename)
        if video_path:
            temp_dir = os.path.join(video_chunk_root, 'temp_imgs')
            img_paths = _extract_frames_to_temp(video_path, temp_dir)
            if img_paths:
                ep_ft_array = sample_images(img_paths)
                temp_video_stats = get_feature_stats_img(ep_ft_array, axis=(0, 2, 3), keepdims=True)
                video_stats = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in temp_video_stats.items()}
                video_stats = {k: v.tolist() for k, v in video_stats.items()}
                episode_dic['stats']['observation.image'] = video_stats
            else:
                print(f"[episodes_stats] Warning: no frames extracted from {video_path}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"[episodes_stats] Video not found for {episode_basename}; skipping observation.image stats")
        # observation.state
        if 'observation.state' in df.columns:
            # Use position-based access to avoid KeyError when index is not RangeIndex
            observation_state_list = [df['observation.state'].iloc[j].tolist() for j in range(len(df['observation.state']))]
            episode_dic['stats']['observation.state'] = get_feature_stats(observation_state_list, axis=0, keepdims=0)
        # action
        if 'action' in df.columns:
            # Use position-based access to avoid KeyError when index is not RangeIndex
            action_list = [df['action'].iloc[j].tolist() for j in range(len(df['action']))]
            episode_dic['stats']['action'] = get_feature_stats(action_list, axis=0, keepdims=0)
        # Scalar numeric columns
        for col in ['episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index']:
            if col in df.columns:
                try:
                    episode_dic['stats'][col] = get_feature_stats(df[col].to_numpy(), axis=0, keepdims=1)
                except Exception:
                    pass
        jsonl_data.append(episode_dic)
    meta_dir = os.path.join(dataset_root, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    out_path = os.path.join(meta_dir, 'episodes_stats.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry))
            f.write('\n')
    print(f"[episodes_stats] Wrote stats for {len(jsonl_data)} episodes -> {out_path}")
    # Optionally refresh info.json minimal fields if present (non-authoritative vs episodes.jsonl)
    try:
        meta_dir = os.path.join(dataset_root, 'meta')
        info_path = os.path.join(meta_dir, 'info.json')
        if os.path.isfile(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f) or {}
        else:
            info = {}
        # Don't overwrite if episodes_jsonl already set; just ensure not missing
        info.setdefault('total_episodes', len(jsonl_data))
        info.setdefault('total_frames', total_frames_accum)
        info.setdefault('total_videos', len(jsonl_data))
        if 'total_chunks' not in info:
            data_dir = os.path.join(dataset_root, 'data')
            chunks = [d for d in os.listdir(data_dir) if d.startswith('chunk') and os.path.isdir(os.path.join(data_dir, d))] if os.path.isdir(data_dir) else []
            info['total_chunks'] = len(chunks)
        os.makedirs(meta_dir, exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"[episodes_stats] info.json touched -> {info_path}")
    except Exception as e:
        print(f"[episodes_stats] info.json update skipped: {e}")
    return out_path

if __name__ == '__main__':
    dataset_root = os.environ.get('DATASET_ROOT', os.path.join(os.environ['HOME'], 'training_data/lerobot/my_pusht'))
    try:
        generate_episodes_stats_jsonl(dataset_root)
    except Exception as e:
        print(f"[episodes_stats] Failed: {e}")
