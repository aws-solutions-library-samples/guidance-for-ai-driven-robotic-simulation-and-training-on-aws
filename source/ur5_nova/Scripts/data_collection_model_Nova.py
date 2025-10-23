import time
from pathlib import Path
import os
import imageio
import numpy
import requests
import torch
import torch.nn as nn
import importlib.util
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
from geometry_msgs.msg import TwistStamped
import threading
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
from cv_bridge import CvBridge
import copy
from math import sin, cos, pi
import math
import gc
import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from lerobot.policies.act.modeling_act import ACTPolicy
import subprocess
import shutil
import inspect
# Nova Pro imports
import boto3
import json
import base64

# Environment variable for asset fetching
os.environ["OMNI_FETCH_ASSETS"] = "1"

# Nova Pro setup
nova_enabled = os.getenv('ENABLE_NOVA_OBSERVER', '0').strip() in ('1', 'true', 'True')
nova_region = os.getenv('AWS_REGION', 'us-east-1')
nova_observation_interval = int(os.getenv('NOVA_OBSERVATION_INTERVAL', '50'))  # steps
bedrock_client = None
if nova_enabled:
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name=nova_region)
        print(f"\033[36m[Nova] Observer enabled, interval: {nova_observation_interval} steps\033[0m")
    except Exception as e:
        print(f"\033[33m[Nova] Failed to initialize: {e}\033[0m")
        nova_enabled = False

bridge = CvBridge()
stop_event = threading.Event()
tool_pose_xy = [0.0, 0.0]
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained model path - prefer host-mounted path inside Docker
def resolve_pretrained_model_path() -> Path:
    env_path = os.getenv("PRETRAINED_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            print(f"\033[36m[Model Path] Using env PRETRAINED_MODEL_PATH: {p}\033[0m")
            return p
    ws_path = Path("/ws/Scripts/pretrained_model")
    if ws_path.exists():
        print(f"\033[36m[Model Path] Using host-mounted pretrained dir: {ws_path}\033[0m")
        return ws_path
    host_fallback = Path(os.path.expanduser("~/ur5_push_T-main/Scripts/pretrained_model"))
    if host_fallback.exists():
        print(f"\033[36m[Model Path] Using fallback local pretrained dir: {host_fallback}\033[0m")
        return host_fallback
    print("\033[33m[Model Path] Warning: pretrained model dir not found; expected at /ws/Scripts/pretrained_model\033[0m")
    return ws_path

pretrained_policy_path = resolve_pretrained_model_path()
# Propagate to env so downstream libs/scripts read the same local directory
os.environ["PRETRAINED_MODEL_PATH"] = str(pretrained_policy_path)

# RL finetune checkpoint directory (same as in 5_Fine.bak)
# Default tries workspace-relative path first for container compatibility
default_checkpoint_dir = "/ws/Scripts/rl_finetune" if os.path.exists("/ws/Scripts") else "~/ur5_push_T-main/Scripts/rl_finetune"
checkpoint_dir = Path(os.getenv("RL_FINETUNE_DIR", default_checkpoint_dir)).expanduser()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
LATEST_CKPT_NAME = "last_checkpoint.pt"

# Optional: explicit path override (kept for backward compatibility). If provided and is a file/dir, will be tried first.
rl_finetune_path_env = os.getenv("RL_FINETUNE_PATH", "").strip()
policy_lock = threading.RLock()

def configure_reproducibility_from_env():
    """Configure seeds and deterministic settings from environment variables.

    ENV:
      - GLOBAL_SEED or SEED: integer seed value. If unset, seeding is skipped.
      - DETERMINISTIC: 1/true to request deterministic algorithms (best effort).
    """
    seed_value = None
    seed_str = os.getenv("GLOBAL_SEED", "").strip() or os.getenv("SEED", "").strip()
    if seed_str:
        try:
            seed_value = int(seed_str)
        except Exception:
            seed_value = None
    deterministic_flag = os.getenv("DETERMINISTIC", "0").strip() in ("1", "true", "True")
    # Set CuBLAS workspace for stricter determinism if requested (best effort)
    if deterministic_flag and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    if seed_value is not None:
        try:
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            print(f"\033[36m[Repro] Seeds set to {seed_value}\033[0m")
        except Exception as e:  # noqa: BLE001
            print(f"\033[33m[Repro] Seed setting warning: {e}\033[0m")
    if deterministic_flag:
        try:
            # Prefer strict deterministic algorithms when available
            torch.use_deterministic_algorithms(True)
        except Exception:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        print("\033[36m[Repro] Deterministic mode requested\033[0m")

def _is_action_deterministic_requested() -> bool:
    return os.getenv('ACTION_DETERMINISTIC', '1').strip() in ('1', 'true', 'True')

def _invoke_select_action(policy_obj, obs, deterministic_requested: bool):
    """Try to call select_action deterministically if the API supports it.

    Tries common keyword names, falls back to default call.
    """
    method = getattr(policy_obj, 'select_action', None)
    if method is None:
        raise AttributeError('Policy has no select_action method')
    if deterministic_requested:
        try:
            sig = inspect.signature(method)
            params = sig.parameters
            for true_kw in ('deterministic', 'use_mean', 'greedy'):
                if true_kw in params:
                    return method(obs, **{true_kw: True})
            for false_kw in ('sample', 'stochastic'):
                if false_kw in params:
                    return method(obs, **{false_kw: False})
        except Exception:
            pass
    return method(obs)

def _try_load_checkpoint_into(policy_obj, path: Path) -> bool:
    """Try load a checkpoint file that contains a 'state_dict'. Returns success bool."""
    try:
        if not path.exists():
            return False
        data = torch.load(path, map_location=device)
        state_dict = data.get("state_dict", data)
        policy_obj.load_state_dict(state_dict)  # type: ignore[arg-type]
        policy_obj.to(device)
        policy_obj.eval()
        tag = data.get("tag", "<no-tag>") if isinstance(data, dict) else "<raw>"
        print(f"\033[35m[RL Finetune] Checkpoint loaded: {path.name} (tag={tag})\033[0m")
        return True
    except Exception as e:
        print(f"\033[31m[RL Finetune] Checkpoint could not be loaded ({path}): {e}\033[0m")
        return False

def load_latest_checkpoint_if_exists(policy_obj) -> bool:
    """Load LATEST_CKPT_NAME or newest policy_*.pt. Return True if success."""
    # 1) Explicit env override file path
    if rl_finetune_path_env:
        alt = Path(rl_finetune_path_env).expanduser()
        if alt.is_file():
            if _try_load_checkpoint_into(policy_obj, alt):
                print("\033[36m[RL Finetune] (ENV path) used.\033[0m")
                return True
        elif alt.is_dir():
            # treat as checkpoint directory
            alt_ckpt = alt / LATEST_CKPT_NAME
            if _try_load_checkpoint_into(policy_obj, alt_ckpt):
                print("\033[36m[RL Finetune] (ENV dir) latest checkpoint loaded.\033[0m")
                return True
    latest_path = checkpoint_dir / LATEST_CKPT_NAME
    if _try_load_checkpoint_into(policy_obj, latest_path):
        return True
    # scan pattern policy_*.pt
    try:
        candidates = sorted(checkpoint_dir.glob("policy_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            if _try_load_checkpoint_into(policy_obj, cand):
                return True
    except Exception as e:
        print(f"\033[31m[RL Finetune] Checkpoint scan error: {e}\033[0m")
    return False

def _load_policy_for_evaluation():
    """
    First loads the pretrained ACT model; if RL finetune checkpoint (last_checkpoint.pt or policy_*.pt) exists
    and RL_FINETUNE_FRESH_START != 1, applies those weights. This way the logic from 5_Fine.bak
    (continue with last finetuned weights) also applies here.
    ENV Variables:
      RL_FINETUNE_FRESH_START=1  -> Skip checkpoint search, use pretrained directly.
      RL_FINETUNE_PATH           -> (optional) Specific file or folder.
      RL_FINETUNE_DIR            -> Checkpoint folder (default ~/lerobot/outputs/rl_finetune)
    """
    fresh_env = os.getenv("RL_FINETUNE_FRESH_START", "0").strip()
    skip_ckpt = fresh_env in ("1", "true", "True")
    base_path = pretrained_policy_path
    print(f"\033[36m[Model Loading] Pretrained model loading: {base_path}\033[0m")
    print(f"\033[36m[Model Loading] Device: {device}\033[0m")
    try:
        p = ACTPolicy.from_pretrained(base_path)
        p.to(device)
        p.eval()
        total_params = sum(param.numel() for param in p.parameters())
        trainable_params = sum(param.numel() for param in p.parameters() if param.requires_grad)
        print(f"\033[32m[Model Loading] ✓ Pretrained model loaded\033[0m")
        print(f"\033[36m[Model Info] Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}\033[0m")
        if not skip_ckpt:
            loaded = load_latest_checkpoint_if_exists(p)
            if loaded:
                print("\033[32m[RL Finetune] ✓ last finetune weights applied\033[0m")
            else:
                print("\033[33m[RL Finetune] Finetune checkpoint is not found, pretrained is used\033[0m")
        else:
            print("\033[33m[RL Finetune] Fresh start; checkpoint is not searched\033[0m")
        if hasattr(p, 'select_action'):
            print("\033[32m[Model Info] ✓ select_action exists\033[0m")
        else:
            print("\033[33m[Model Info] ⚠ select_action does not exist\033[0m")
        return p
    except Exception as e:
        print(f"\033[31m[Model Loading] ✗ Model could not be loaded: {e}\033[0m")
        raise

def _load_policy_from_path_str(path_str: str):
    print(f"\033[36m[Alt Model Loading] Loading alternate model from: {path_str}\033[0m")
    try:
        p = ACTPolicy.from_pretrained(Path(path_str).expanduser())
        p.to(device)
        p.eval()
        total_params = sum(p.numel() for p in p.parameters())
        trainable_params = sum(p.numel() for p in p.parameters() if p.requires_grad)
        print(f"\033[32m[Alt Model Loading] ✓ Alt model loaded successfully\033[0m")
        print(f"\033[32m[Alt Model Loading] ✓ Alt model moved to {device}\033[0m")
        print(f"\033[32m[Alt Model Loading] ✓ Alt model set to eval mode\033[0m")
        print(f"\033[36m[Alt Model Info] Total parameters: {total_params:,}\033[0m")
        print(f"\033[36m[Alt Model Info] Trainable parameters: {trainable_params:,}\033[0m")
        return p
    except Exception as e:
        print(f"\033[31m[Alt Model Loading] ✗ Failed to load alt model: {e}\033[0m")
        return None

configure_reproducibility_from_env()
policy = _load_policy_for_evaluation()
alt_model_env = os.getenv("ALT_MODEL_PATH", "").strip()
alt_policy = _load_policy_from_path_str(alt_model_env) if alt_model_env else None

def reload_models_after_reset():
    """Reload primary (and alternate if defined) policies after a reset.

    Ensures thread safety with policy_lock so inference threads don't race.
    Called after every simulation reset per user request.
    """
    global policy, alt_policy
    try:
        with policy_lock:
            # Re-resolve pretrained path on every reset to pick up host updates
            global pretrained_policy_path
            pretrained_policy_path = resolve_pretrained_model_path()
            os.environ["PRETRAINED_MODEL_PATH"] = str(pretrained_policy_path)
            print("\033[36m[Model Reload] Reloading primary policy...\033[0m")
            policy = _load_policy_for_evaluation()
            alt_model_env_local = os.getenv("ALT_MODEL_PATH", "").strip()
            if alt_model_env_local:
                print("\033[36m[Model Reload] Reloading alternate policy...\033[0m")
                alt_policy = _load_policy_from_path_str(alt_model_env_local)
            else:
                alt_policy = None
            print("\033[32m[Model Reload] Done.\033[0m")
    except Exception as e:  # noqa: BLE001
        print(f"\033[31m[Model Reload] Failed: {e}\033[0m")

class Get_End_Effector_Pose(Node):
    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            TFMessage,
            '/isaac_tf',
            self.listener_callback,
            10)
        self.euler_angles = np.array([0.0, 0.0, 0.0], float)

    def listener_callback(self, data):
        global tool_pose_xy, tbar_pose_xyw
        tool_pose = data.transforms[0].transform.translation
        tool_pose_xy[0] = tool_pose.y
        tool_pose_xy[1] = tool_pose.x
        tbar_translation = data.transforms[1].transform.translation
        tbar_rotation = data.transforms[1].transform.rotation
        tbar_pose_xyw[0] = tbar_translation.y
        tbar_pose_xyw[1] = tbar_translation.x
        self.euler_angles[:] = R.from_quat([tbar_rotation.x, tbar_rotation.y, tbar_rotation.z, tbar_rotation.w]).as_euler('xyz', degrees=False)
        tbar_pose_xyw[2] = self.euler_angles[2]

class RobotHomeCommander(Node):
    """Persistent publisher to reliably send the robot to a home pose.
    Transient node publish can get dropped due to discovery delay; this keeps a latched publisher alive."""
    def __init__(self, topic:str = '/arm_controller/joint_trajectory'):
        super().__init__('robot_home_commander')
        self._topic = topic
        self._pub = self.create_publisher(JointTrajectory, topic, 10)
        # default joint order (adjust if your controller expects different names)
        self._joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

    def send_home(self, positions=None, time_to_reach: float = 3.0, repeat: int = 3):
        if positions is None:
            positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        traj = JointTrajectory()
        traj.joint_names = list(self._joint_names)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        # leaving velocities empty can let controller compute; provide zeros for clarity
        point.velocities = [0.0]*len(positions)
        # Fill time_from_start
        secs = int(time_to_reach)
        nanos = int((time_to_reach - secs)*1e9)
        point.time_from_start.sec = secs
        point.time_from_start.nanosec = nanos
        traj.points = [point]
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = 'base'
        for i in range(max(1, repeat)):
            self._pub.publish(traj)
            # tiny sleep to allow multiple sends, improves reliability with some controllers
            time.sleep(0.05)
        print(f"\033[36m[RobotHomeCommander] Home traj published ({repeat}x) to {self._topic}\033[0m")

# Global reference (assigned in main)
robot_home_commander: RobotHomeCommander | None = None

def sync_dataset_from_host(host_dataset_path: str, container_dataset_path: str):
    """
    Checks existing dataset on host and adjusts container's episode index accordingly.
    
    Args:
        host_dataset_path: Dataset path on host (shared via volume mount)
        container_dataset_path: Dataset path inside container (e.g.: /ws/Scripts/dataset/)
    
    This function is called at program start and:
    1. Detects existing episodes on host
    2. Adjusts container's episode index based on max episode on host
    3. Data is already shared via volume mount, no explicit copying needed
    
    Returns:
        dict: {'max_episode': int, 'max_data_index': int, 'episode_count': int}
    """
    result = {'max_episode': 0, 'max_data_index': 0, 'episode_count': 0}
    
    try:
        print("\033[36m" + "="*60 + "\033[0m")
        print("\033[36m[Sync Init] Host → Container Dataset Synchronization\033[0m")
        print("\033[36m" + "="*60 + "\033[0m")
        
        # Host path'i expand et
        host_path = Path(host_dataset_path).expanduser()
        container_path = Path(container_dataset_path)
        
        # Check if dataset exists on host
        if not host_path.exists():
            print(f"\033[33m[Sync Init] ⚠️  Host dataset folder not found: {host_path}\033[0m")
            print(f"\033[33m[Sync Init] New dataset folder will be created\033[0m")
            return result
        
        print(f"\033[32m[Sync Init] ✓ Host dataset found: {host_path}\033[0m")
        
        # Scan episodes on host
        host_data_dir = host_path / "data" / "chunk-000"
        if not host_data_dir.exists():
            print(f"\033[33m[Sync Init] No data folder on host yet, fresh start\033[0m")
            return result
        
        # Find episode files
        parquet_files = list(host_data_dir.glob("episode_*.parquet"))
        result['episode_count'] = len(parquet_files)
        
        if not parquet_files:
            print(f"\033[33m[Sync Init] No episode files found on host, fresh start\033[0m")
            return result
        
        # Extract episode indices
        episode_indices = sorted([int(f.stem.split('_')[1]) for f in parquet_files])
        result['max_episode'] = max(episode_indices)
        
        print(f"\033[32m[Sync Init] ✓ Found {len(parquet_files)} episodes on host\033[0m")
        print(f"\033[36m[Sync Init] Episode range: {min(episode_indices)} - {max(episode_indices)}\033[0m")
        print(f"\033[35m[Sync Init] Latest episode: episode_{max(episode_indices):06d}.parquet\033[0m")
        
        # Calculate global max data index (sum of frames in all episodes)
        total_frames = 0
        max_global_index = 0
        
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                if 'index' in df.columns and len(df) > 0:
                    file_max = df['index'].max()
                    max_global_index = max(max_global_index, file_max)
                    total_frames += len(df)
            except Exception as e:
                print(f"\033[33m[Sync Init] ⚠️  {parquet_file.name} could not be read: {e}\033[0m")
        
        result['max_data_index'] = max_global_index
        
        print(f"\033[36m[Sync Init] Total frame count: {total_frames}\033[0m")
        print(f"\033[36m[Sync Init] Max global data index: {max_global_index}\033[0m")
        
        # Check video files
        video_dir = host_path / "videos" / "chunk-000" / "observation.images.state"
        if video_dir.exists():
            video_files = list(video_dir.glob("episode_*.mp4"))
            print(f"\033[32m[Sync Init] ✓ Found {len(video_files)} video files on host\033[0m")
        
        # Check meta files
        meta_dir = host_path / "meta"
        if meta_dir.exists():
            jsonl_files = list(meta_dir.glob("*.jsonl"))
            if jsonl_files:
                print(f"\033[32m[Sync Init] ✓ Found {len(jsonl_files)} meta files on host\033[0m")
                for jsonl in jsonl_files:
                    print(f"\033[36m[Sync Init]   - {jsonl.name}\033[0m")
        
        # Volume mount check
        print(f"\n\033[36m[Sync Init] Volume Mount Status:\033[0m")
        if container_path.exists():
            container_data_dir = container_path / "data" / "chunk-000"
            container_episodes = list(container_data_dir.glob("episode_*.parquet")) if container_data_dir.exists() else []
            if len(container_episodes) == len(parquet_files):
                print(f"\033[32m[Sync Init] ✓ Container and Host see same number of episodes ({len(container_episodes)})\033[0m")
                print(f"\033[32m[Sync Init] ✓ Volume mount working correctly\033[0m")
            else:
                print(f"\033[33m[Sync Init] ⚠️  Container: {len(container_episodes)} episodes, Host: {len(parquet_files)} episodes\033[0m")
        
        print(f"\n\033[32m[Sync Init] ✓ Dataset synchronization completed\033[0m")
        print(f"\033[35m[Sync Init] 📝 New episode starting index: {result['max_episode'] + 1}\033[0m")
        print(f"\033[35m[Sync Init] 📝 New data index starting: {result['max_data_index'] + 1}\033[0m")
        print("\033[36m" + "="*60 + "\033[0m\n")
        
    except Exception as e:
        print(f"\033[31m[Sync Init] ❌ Host dataset check error: {e}\033[0m")
        import traceback
        traceback.print_exc()
    
    return result

def sync_dataset_to_host(container_dataset_path: str, host_dataset_path: str):
    """
    Synchronizes dataset data from container to host.
    
    Args:
        container_dataset_path: Dataset path inside container (e.g.: /ws/Scripts/dataset/)
        host_dataset_path: Dataset path on host (shared via volume mount)
    
    This function:
    1. Syncs filesystem (flushes buffers)
    2. Optionally performs explicit copying with rsync
    """
    try:
        print("\033[36m[Sync] Starting dataset synchronization...\033[0m")
        
        # 1. Sync filesystem (flush all buffers to disk)
        try:
            subprocess.run(['sync'], check=False, timeout=10)
            print("\033[32m[Sync] ✓ Filesystem sync completed\033[0m")
        except Exception as e:
            print(f"\033[33m[Sync] Filesystem sync warning: {e}\033[0m")
        
        # 2. Since volume mount is used, explicit copy may not be needed,
        #    but let's verify files are written for safety
        container_path = Path(container_dataset_path)
        if container_path.exists():
            # Check parquet files in data/ folder
            data_dir = container_path / "data" / "chunk-000"
            if data_dir.exists():
                parquet_files = list(data_dir.glob("episode_*.parquet"))
                print(f"\033[32m[Sync] ✓ {len(parquet_files)} parquet files available\033[0m")
            
            # Check video files in videos/ folder
            video_dir = container_path / "videos" / "chunk-000" / "observation.images.state"
            if video_dir.exists():
                video_files = list(video_dir.glob("episode_*.mp4"))
                print(f"\033[32m[Sync] ✓ {len(video_files)} video files available\033[0m")
            
            # Check jsonl files in meta/ folder
            meta_dir = container_path / "meta"
            if meta_dir.exists():
                jsonl_files = list(meta_dir.glob("*.jsonl"))
                print(f"\033[32m[Sync] ✓ {len(jsonl_files)} meta files available\033[0m")
        
        # 3. Optional rsync can be activated via ENV variable
        enable_rsync = os.getenv('ENABLE_RSYNC_SYNC', '0').strip() in ('1', 'true', 'True')
        if enable_rsync and host_dataset_path:
            try:
                # Explicit copying with rsync (if rsync is installed)
                host_path = Path(host_dataset_path).expanduser()
                cmd = [
                    'rsync', '-avz', '--delete',
                    f'{container_dataset_path}/',
                    f'{host_path}/'
                ]
                result = subprocess.run(cmd, check=True, timeout=60, capture_output=True, text=True)
                print(f"\033[32m[Sync] ✓ rsync completed: {container_dataset_path} → {host_path}\033[0m")
            except FileNotFoundError:
                print("\033[33m[Sync] rsync not found, relying on volume mount\033[0m")
            except subprocess.TimeoutExpired:
                print("\033[31m[Sync] rsync timeout (60s exceeded)\033[0m")
            except Exception as e:
                print(f"\033[31m[Sync] rsync error: {e}\033[0m")
        
        print("\033[32m[Sync] ✓ Dataset synchronization completed\033[0m")
        
    except Exception as e:
        print(f"\033[31m[Sync] Synchronization error: {e}\033[0m")

def _load_sibling_func(module_filename: str, func_name: str):
    """Safely load a function from a sibling python file using its path.

    Works even when this script is executed directly (no package context)."""
    try:
        mod_path = Path(__file__).parent / module_filename
        if not mod_path.exists():
            raise FileNotFoundError(f"{mod_path} not found")
        spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"spec load failed for {mod_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return getattr(module, func_name)
    except Exception as e:  # noqa: BLE001
        print(f"\033[33m[MetaGen] Could not load {func_name} from {module_filename}: {e}\033[0m")
        return None

def generate_episode_meta_jsonl(dataset_root: str):
    """Generate or update episodes.jsonl and episodes_stats.jsonl under dataset_root."""
    _gen_ep = _load_sibling_func('create_episodes_jsonl.py', 'generate_episodes_jsonl')
    _gen_stats = _load_sibling_func('create_episodes_stats_jsonl.py', 'generate_episodes_stats_jsonl')
    if callable(_gen_ep):
        try:
            ep_path = _gen_ep(dataset_root)
            print(f"\033[32m[MetaGen] Updated {ep_path}\033[0m")
        except Exception as e:  # noqa: BLE001
            print(f"\033[33m[MetaGen] episodes.jsonl generation failed: {e}\033[0m")
    if callable(_gen_stats):
        try:
            stats_path = _gen_stats(dataset_root)
            print(f"\033[32m[MetaGen] Updated {stats_path}\033[0m")
        except Exception as e:  # noqa: BLE001
            print(f"\033[33m[MetaGen] episodes_stats.jsonl generation failed: {e}\033[0m")


def _run_repair_index_script(dataset_path: str):
    """Run repair_index.py before meta generation for successful episodes.

    Args:
        dataset_path: Path to the dataset data/chunk-000 directory
    """
    try:
        candidates = []
        try:
            # repo root relative to this file
            repo_root = (Path(__file__).resolve().parent.parent)
            candidates.append(repo_root / 'lerobot_related' / 'repair_index.py')
        except Exception:
            pass
        # common absolute fallbacks
        candidates.append(Path('/ws/lerobot_related/repair_index.py'))
        candidates.append(Path(os.path.expanduser('~/ur5_push_T-main/lerobot_related/repair_index.py')))
        script_path = None
        for p in candidates:
            if p.exists():
                script_path = p
                break
        if script_path is None:
            print("\033[33m[RepairIndex] Script not found; skipping.\033[0m")
            return
        print(f"\033[36m[RepairIndex] Running: {script_path} with dataset: {dataset_path}\033[0m")
        # Use same python executable environment and pass dataset path as argument
        subprocess.run([shutil.which('python3') or 'python3', str(script_path), dataset_path], check=True)
        print("\033[32m[RepairIndex] ✓ Completed\033[0m")
    except subprocess.CalledProcessError as e:
        print(f"\033[31m[RepairIndex] Script failed: {e}\033[0m")
    except Exception as e:  # noqa: BLE001
        print(f"\033[33m[RepairIndex] Could not run script: {e}\033[0m")

def nova_observe_scene(image, robot_pos, tbar_pos, accuracy, step_count, episode_index):
    """Use Nova Pro to observe and comment on the current scene"""
    if not nova_enabled or bedrock_client is None:
        return
    
    try:
        # Encode image for Nova Pro
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""You are observing a UR5 robot arm performing a T-bar pushing task. 

CURRENT STATUS:
- Episode: {episode_index}, Step: {step_count}
- Robot position: {robot_pos}
- T-bar position: {tbar_pos}
- Task accuracy: {accuracy:.3f}
- Target: Push T-bar toward center [0.0, 0.0]

Please provide a brief observation about:
1. Robot's current approach/strategy
2. T-bar positioning relative to target
3. Any notable behaviors or potential improvements

Keep response under 100 words."""
        
        response = bedrock_client.invoke_model(
            modelId='amazon.nova-pro-v1:0',
            body=json.dumps({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {"image": {"format": "jpeg", "source": {"bytes": image_b64}}}
                    ]
                }],
                "inferenceConfig": {
                    "maxTokens": 150,
                    "temperature": 0.3
                }
            })
        )
        
        result = json.loads(response['body'].read())
        observation = result['output']['message']['content'][0]['text']
        
        print(f"\033[35m[Nova Observer] {observation}\033[0m")
        
    except Exception as e:
        print(f"\033[33m[Nova] Observation failed: {e}\033[0m")

class Action_Publisher(Node):
    def __init__(self):
        super().__init__('Joy_Publisher')
        self.declare_parameter('hz', 10)
        self.declare_parameter('control_mode', 'joy')  # 'joy' or 'twist'
        self.declare_parameter('twist_topic', '/servo_node/delta_twist_cmds')
        self.declare_parameter('success_threshold', 0.80)
        self.declare_parameter('max_episode_steps', 600)
        self.declare_parameter('accuracy_floor', 0.17)
        self.declare_parameter('peak_min', 0.50)
        self.declare_parameter('drop_threshold', 0.35)
        self.declare_parameter('floor_warmup_steps', 30)
        self.declare_parameter('diagnostics', False)
        # Default output_dir: container-aware (prefers /ws if exists)
        default_output_dir = "/ws/Scripts/dataset/" if os.path.exists("/ws/Scripts") else os.environ["HOME"] + "/ur5_push_T-main/Scripts/dataset/"
        self.declare_parameter('output_dir', default_output_dir)
        self.enable_data_collection = os.getenv('ENABLE_DATA_COLLECTION', '1').strip() in ('1', 'true', 'True')
        Hz = int(self.get_parameter('hz').get_parameter_value().integer_value)
        self.success_threshold = float(self.get_parameter('success_threshold').get_parameter_value().double_value or 0.90)
        self.max_episode_steps = int(self.get_parameter('max_episode_steps').get_parameter_value().integer_value or 600)
        self.accuracy_floor = float(self.get_parameter('accuracy_floor').get_parameter_value().double_value or 0.21)
        self.peak_min = float(self.get_parameter('peak_min').get_parameter_value().double_value or 0.50)
        self.drop_threshold = float(self.get_parameter('drop_threshold').get_parameter_value().double_value or 0.23)
        self.floor_warmup_steps = int(self.get_parameter('floor_warmup_steps').get_parameter_value().integer_value or 30)
        diag_env = os.getenv('DIAG', '0').strip()
        self.diagnostics = bool(self.get_parameter('diagnostics').get_parameter_value().bool_value or (diag_env in ('1','true','True')))
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        
        # Host dataset path for sync (defaults to host workspace location)
        self.host_dataset_path = os.getenv('HOST_DATASET_PATH', os.path.expanduser('~/ur5_push_T-main/Scripts/dataset/')).rstrip('/')
        
        # Sync dataset from host at startup (check existing data and update indices)
        if self.enable_data_collection:
            sync_result = sync_dataset_from_host(self.host_dataset_path, self.output_dir)
            # Store sync result for later use if needed
            self._sync_result = sync_result
        
        self.pub_joy = self.create_publisher(Joy, '/joy', 10)
        twist_topic = self.get_parameter('twist_topic').get_parameter_value().string_value or '/servo_node/delta_twist_cmds'
        self.pub_twist = self.create_publisher(TwistStamped, twist_topic, 10)
        self.joy_commands = Joy()
        self.joy_commands.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.joy_commands.buttons = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.timer = self.create_timer(1/Hz, self.timer_callback)
        self.control_mode = (self.get_parameter('control_mode').get_parameter_value().string_value or 'joy').strip()
        
        # Image paths: container-aware (prefers /ws if exists)
        images_dir = "/ws/images" if os.path.exists("/ws/images") else os.environ['HOME'] + "/ur5_push_T-main/images"
        self.initial_image = cv2.imread(os.path.join(images_dir, "stand_top_plane.png"))
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.tool_radius = 10
        self.scale = 1.639344
        self.C_W = 182
        self.C_H = 152
        self.OBL1 = int(150/self.scale)
        self.OBL2 = int(120/self.scale)
        self.OBW = int(30/self.scale)
        self.radius = int(10/self.scale)
        self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)
        self.T_image = cv2.imread(os.path.join(images_dir, "stand_top_plane_filled.png"))
        self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
        thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        self.blue_region = cv2.bitwise_not(img_th)
        self.blue_region_sum = cv2.countNonZero(self.blue_region)
        self.step_count = 0
        self.episode_index = self._get_next_episode_index()
        self.best_accuracy = 0.0
        self.Hz = 10
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.start_recording = False
        self.data_recorded = False
        if self.enable_data_collection:
            self.log_dir = os.path.join(self.output_dir, "data/chunk-000/")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            base_vid_dir = os.path.join(self.output_dir, 'videos/chunk-000/observation.images.')
            self.state_vid_dir = base_vid_dir + 'state/'
            if not os.path.exists(self.state_vid_dir):
                os.makedirs(self.state_vid_dir)
            self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index'])
            self.index = self._get_next_data_index()
            self.frame_index = 0
            self.time_stamp = 0.0
            self.success = False
            self.done = False
            self.column_index = 0
            self.prev_sum = 0.0
            self.state_image_array = []
        self.act_hist = []
        self.alt_act_hist = []
        # Recording delay (seconds) after each reset before frames are logged
        self.record_delay_secs = float(os.getenv('RECORD_DELAY_SECS', '1.0'))
        self.record_resume_time = time.time() + self.record_delay_secs
        # Nova observation tracking
        self.last_nova_observation = 0

    def _get_next_episode_index(self):
        if not self.enable_data_collection:
            return 0
        log_dir = os.path.join(self.output_dir, "data/chunk-000/")
        if not os.path.exists(log_dir):
            return 0
        episode_files = [f for f in os.listdir(log_dir) if f.startswith('episode_') and f.endswith('.parquet')]
        if not episode_files:
            return 0
        indices = [int(f.split('_')[1].split('.')[0]) for f in episode_files]
        return max(indices) + 1 if indices else 0

    def _get_next_data_index(self):
        if not self.enable_data_collection:
            return 0
        log_dir = os.path.join(self.output_dir, "data/chunk-000/")
        if not os.path.exists(log_dir):
            return 0
        episode_files = [f for f in os.listdir(log_dir) if f.startswith('episode_') and f.endswith('.parquet')]
        if not episode_files:
            return 0
        max_index = 0
        for f in episode_files:
            try:
                df = pd.read_parquet(os.path.join(log_dir, f))
                if 'index' in df.columns:
                    max_index = max(max_index, df['index'].max())
            except Exception:
                continue
        return max_index + 1

    def _stop_and_reset(self, reason: str):
        print('\033[33m' + f"Stopping episode due to: {reason}" + '\033[0m')
        if robot_home_commander:
            robot_home_commander.send_home()
        time.sleep(3.0)
        self.reset_simulation()
        if self.enable_data_collection:
            self.df = self.df.iloc[0:0]
            self.state_image_array.clear()
        self.step_count = 0
        self.best_accuracy = 0.0
        self.act_hist.clear()
        self.alt_act_hist.clear()
        # Set new resume time for recording (enforce delay after reset)
        self.record_resume_time = time.time() + self.record_delay_secs
        # Reset Nova observation counter
        self.last_nova_observation = 0
        print(f"\033[36mStarting new episode (evaluation): {self.episode_index}\033[0m")

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw
        self.joy_commands.header.frame_id = "joy"
        self.joy_commands.header.stamp = self.get_clock().now().to_msg()
        base_image = copy.copy(self.initial_image)
        self.Tbar_region[:] = 0
        x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
        y = int((tool_pose_xy[1]*1000 - 320)/self.scale)
        cv2.circle(base_image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)
        x1 = tbar_pose_xyw[0]
        y1 = tbar_pose_xyw[1]
        th1 = -tbar_pose_xyw[2] - pi/2
        dx1 = -self.OBW/2*cos(th1 - pi/2)
        dy1 = -self.OBW/2*sin(th1 - pi/2)
        self.tbar1_ob = [[int(cos(th1)*self.OBL1/2     - sin(th1)*self.OBW/2   + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2    + cos(th1)*self.OBW/2   + dy1 + (1000*y1-320)/self.scale)],
                         [int(cos(th1)*self.OBL1/2    - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*self.OBL1/2    + cos(th1)*(-self.OBW/2)+ dy1 + (1000*y1-320)/self.scale)],
                         [int(cos(th1)*(-self.OBL1/2) - sin(th1)*(-self.OBW/2)+ dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*(-self.OBW/2)+ dy1 + (1000*y1-320)/self.scale)],
                         [int(cos(th1)*(-self.OBL1/2) - sin(th1)*self.OBW/2   + dx1 + self.C_W + 1000*x1/self.scale), int(sin(th1)*(-self.OBL1/2) + cos(th1)*self.OBW/2   + dy1 + (1000*y1-320)/self.scale)]]
        pts1_ob = np.array(self.tbar1_ob, np.int32)
        cv2.fillPoly(base_image, [pts1_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)
        th2 = -tbar_pose_xyw[2] - pi
        dx2 = self.OBL2/2*cos(th2)
        dy2 = self.OBL2/2*sin(th2)
        self.tbar2_ob = [[int(cos(th2)*self.OBL2/2    - sin(th2)*self.OBW/2    + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)],
                         [int(cos(th2)*self.OBL2/2    - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                         [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                         [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2   + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)]]
        pts2_ob = np.array(self.tbar2_ob, np.int32)
        cv2.fillPoly(base_image, [pts2_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)
        cv2.circle(base_image, center=(int(self.C_W + 1000*x1/self.scale), int((1000*y1-320)/self.scale)), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)
        img_msg = bridge.cv2_to_imgmsg(base_image)
        self.pub_img.publish(img_msg)
        common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
        common_part_sum = cv2.countNonZero(common_part)
        accuracy = common_part_sum/self.blue_region_sum if self.blue_region_sum > 0 else 0.0
        if self.step_count % 10 == 0:
            print(f"step {self.step_count} | ep {self.episode_index} | accuracy: {accuracy:.3f}")
        
        # Nova Pro scene observation
        if nova_enabled and (self.step_count - self.last_nova_observation) >= nova_observation_interval:
            nova_observe_scene(base_image, tool_pose_xy, tbar_pose_xyw[:2], accuracy, self.step_count, self.episode_index)
            self.last_nova_observation = self.step_count
        state_t = torch.from_numpy(np.array(tool_pose_xy)).to(torch.float32).to(device).unsqueeze(0)
        image_t = torch.from_numpy(base_image).to(torch.float32) / 255
        image_t = image_t.permute(2, 0, 1).to(device).unsqueeze(0)
        obs_t = {
            "observation.state": state_t,
            "observation.image": image_t,
        }
        try:
            assert state_t.shape == (1, 2), f'state shape {state_t.shape} != (1,2)'
            assert state_t.dtype == torch.float32, f'state dtype {state_t.dtype} != float32'
            assert image_t.ndim == 4 and image_t.shape[0] == 1 and image_t.shape[1] == 3, f'image shape {image_t.shape} invalid'
            assert image_t.dtype == torch.float32, f'image dtype {image_t.dtype} != float32'
            if torch.isnan(state_t).any() or torch.isinf(state_t).any():
                raise ValueError('state contains NaN/Inf')
            if torch.isnan(image_t).any() or torch.isinf(image_t).any():
                raise ValueError('image contains NaN/Inf')
        except Exception as e:
            print(f"␛[31m[Input Check] Invalid inputs: {e}␛[0m")
            print(f"[Input Dump] state shape={tuple(state_t.shape)} dtype={state_t.dtype} device={state_t.device}")
            print(f"[Input Dump] image shape={tuple(image_t.shape)} dtype={image_t.dtype} device={image_t.device}")
            return
        try:
            with torch.no_grad():
                with policy_lock:
                    action_t = _invoke_select_action(policy, obs_t, _is_action_deterministic_requested())
        except Exception as e:
            print(f"␛[31m[Inference Error] select_action failed: {e}␛[0m")
            print(f"[Obs Dump] keys={list(obs_t.keys())}")
            for k,v in obs_t.items():
                try:
                    print(f"  - {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                except Exception:
                    print("  - {k}: <unprintable>")
            return
        numpy_action = action_t.squeeze(0).to("cpu").numpy()
        numpy_action = np.clip(numpy_action, -1.0, 1.0)
        if self.step_count % 10 == 0:
            print(f"[Action] step {self.step_count} | a=({numpy_action[0]:+.3f},{numpy_action[1]:+.3f})")
        if self.diagnostics:
            alt_numpy_action = None
            if alt_policy is not None:
                try:
                    with torch.no_grad():
                        with policy_lock:
                            alt_out = _invoke_select_action(alt_policy, obs_t, _is_action_deterministic_requested())
                    alt_numpy_action = alt_out.squeeze(0).to("cpu").numpy()
                    alt_numpy_action = np.clip(alt_numpy_action, -1.0, 1.0)
                except Exception as e:
                    print(f"Alt policy select_action failed: {e}")
            self.act_hist.append(numpy_action.copy())
            if alt_numpy_action is not None:
                self.alt_act_hist.append(alt_numpy_action.copy())
            if len(self.act_hist) > 100:
                self.act_hist = self.act_hist[-100:]
            if len(self.alt_act_hist) > 100:
                self.alt_act_hist = self.alt_act_hist[-100:]
            if self.step_count % 10 == 0:
                try:
                    arr = np.array(self.act_hist, dtype=np.float32)
                    var_x = float(arr[:,0].var()) if arr.size else 0.0
                    var_y = float(arr[:,1].var()) if arr.size else 0.0
                    msg = f"primary a=({numpy_action[0]:+.3f},{numpy_action[1]:+.3f}) var=({var_x:.4f},{var_y:.4f})"
                    if alt_numpy_action is not None:
                        arr_alt = np.array(self.alt_act_hist, dtype=np.float32) if self.alt_act_hist else None
                        var_x_alt = float(arr_alt[:,0].var()) if arr_alt is not None and arr_alt.size else 0.0
                        var_y_alt = float(arr_alt[:,1].var()) if arr_alt is not None and arr_alt.size else 0.0
                        msg += f" | alt a=({alt_numpy_action[0]:+.3f},{alt_numpy_action[1]:+.3f}) var=({var_x_alt:.4f},{var_y_alt:.4f})"
                    print(msg)
                    if var_y < 1e-4 and var_x > 1e-3:
                        print("\033[33mWarning: Y-axis action variance near zero; model may be stuck moving mostly along X.\033[0m")
                except Exception:
                    pass
        if self.enable_data_collection:
            if time.time() < self.record_resume_time:
                # Still in delay window; skip recording frames but allow control to run
                if self.step_count % 10 == 0:
                    remaining = self.record_resume_time - time.time()
                    if remaining < 0: remaining = 0
                    print(f"\033[34m[Delay] Recording will start in {remaining:.2f}s (ep {self.episode_index})\033[0m")
            else:
                if accuracy >= self.success_threshold:
                    self.success = True
                    self.done = True
                    print('\033[31m'+'SUCCESS!'+f': {accuracy:.3f}'+'\033[0m')
                else:
                    self.success = False
                print('\033[32m'+f'RECORDING episode:{self.episode_index}, index:{self.index} sum:{accuracy:.3f}'+'\033[0m')
                self.df.loc[self.column_index] = [copy.copy(tool_pose_xy), copy.copy(numpy_action), self.episode_index, self.frame_index, self.time_stamp, accuracy, self.done, self.success, self.index, 0]
                self.column_index += 1
                self.frame_index += 1
                self.time_stamp += 1/self.Hz
                self.index += 1
                self.start_recording = True
                self.state_image_array.append(base_image)
        if self.control_mode == 'twist':
            twist = TwistStamped()
            twist.header.stamp = self.get_clock().now().to_msg()
            twist.header.frame_id = 'base_link'
            # Map to match JoyToServo mapping: axes[0]->linear.y, axes[1]->linear.x
            twist.twist.linear.x = float(numpy_action[1])
            twist.twist.linear.y = float(numpy_action[0])
            twist.twist.linear.z = 0.0
            twist.twist.angular.x = 0.0
            twist.twist.angular.y = 0.0
            twist.twist.angular.z = 0.0
            self.pub_twist.publish(twist)
            if self.step_count % 10 == 0:
                print(f"[Twist] lin=({twist.twist.linear.x:+.3f},{twist.twist.linear.y:+.3f},0.000)")
        else:
            self.joy_commands.axes[0] = float(numpy_action[0])
            self.joy_commands.axes[1] = float(numpy_action[1])
            self.pub_joy.publish(self.joy_commands)
        self.step_count += 1
        if self.step_count >= self.floor_warmup_steps and accuracy < self.accuracy_floor:
            self._stop_and_reset(reason=f"floor_{self.accuracy_floor:.2f}")
            return
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        if self.best_accuracy >= self.peak_min and (self.best_accuracy - accuracy) >= self.drop_threshold:
            self._stop_and_reset(reason=f"drop_{self.drop_threshold:.2f}_from_{self.best_accuracy:.3f}_to_{accuracy:.3f}")
            return
        if self.enable_data_collection and self.start_recording and not self.data_recorded and self.success:
            print('\033[31m'+'WRITING A PARQUET FILE'+'\033[0m')
            data_file_name = f'episode_{str(self.episode_index).zfill(6)}.parquet'
            video_file_name = f'episode_{str(self.episode_index).zfill(6)}.mp4'
            
            # Write and flush parquet file
            parquet_path = os.path.join(self.log_dir, data_file_name)
            table = pa.Table.from_pandas(self.df)
            pq.write_table(table, parquet_path)
            # Explicit flush for parquet file
            try:
                with open(parquet_path, 'rb') as f:
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"\033[33m[Sync] Parquet fsync warning: {e}\033[0m")
            print("The parquet file is generated!")
            
            # Write and flush video file
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_path = os.path.join(self.state_vid_dir, video_file_name)
            # Wrist & top videos removed per requirement; only state video retained
            out3 = cv2.VideoWriter(video_path, fourcc, self.Hz, (self.initial_image.shape[1], self.initial_image.shape[0]))
            for frame3 in self.state_image_array:
                out3.write(frame3)
            out3.release()
            # Explicit flush for video file
            try:
                with open(video_path, 'rb') as f:
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"\033[33m[Sync] Video fsync warning: {e}\033[0m")
            print("The state video is generated!")
            # --- Meta generation (episodes.jsonl & episodes_stats.jsonl) ---
            # Run index repair before meta generation on successful episodes
            _run_repair_index_script(self.log_dir)
            dataset_root_local = os.environ.get('DATASET_ROOT', self.output_dir.rstrip('/'))
            generate_episode_meta_jsonl(dataset_root_local)
            
            # All files written, now synchronize to host
            sync_dataset_to_host(self.output_dir, self.host_dataset_path)
            
            self.data_recorded = True
            self.episode_index += 1  # Increment index after successful episode recording
            self.frame_index = 0
            self.time_stamp = 0.0
            self.column_index = 0
            self.start_recording = False
            self.success = False
            self.done = False
            self.df = self.df.iloc[0:0]
            self.state_image_array.clear()
            print(f"🔄 Ready for episode {self.episode_index}")
            self.data_recorded = False
        done = (accuracy >= self.success_threshold) or (self.step_count >= self.max_episode_steps)
        if done:
            status = 'SUCCESS' if accuracy >= self.success_threshold else 'TIMEOUT'
            print(f"\033[32m{status}!\033[0m")
            if robot_home_commander:
                robot_home_commander.send_home()
            time.sleep(3.0)
            self.reset_simulation()
            # Run meta update at episode end only for successful episodes
            if self.enable_data_collection and self.success:
                # Run index repair before meta generation on successful episodes
                _run_repair_index_script(self.log_dir)
                dataset_root_local = os.environ.get('DATASET_ROOT', self.output_dir.rstrip('/'))
                generate_episode_meta_jsonl(dataset_root_local)
            if self.enable_data_collection and not self.success:
                self.df = self.df.iloc[0:0]
                self.state_image_array.clear()
            self.step_count = 0
            if self.success:
                self.episode_index += 1  # Increment index if successful
            self.best_accuracy = 0.0
            self.act_hist.clear()
            self.alt_act_hist.clear()
            # Apply delay for next episode recording start
            self.record_resume_time = time.time() + self.record_delay_secs
            # Reset Nova observation counter for new episode
            self.last_nova_observation = 0
            print(f"\033[36mStarting new episode (evaluation): {self.episode_index}\033[0m")

    def reset_simulation(self):
        url = "http://127.0.0.1:8099/reset?reload=1"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("\033[32mReset request queued.\033[0m")
            else:
                print(f"\033[31mReset request failed: {response.status_code} {response.text}\033[0m")
        except Exception as e:
            print(f"\033[31mReset request could not be sent: {e}\033[0m")
        # Reload models after each reset (user request)
        reload_models_after_reset()

class Wrist_Camera_Subscriber(Node):
    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_wrist',
            self.camera_callback,
            10)

    def camera_callback(self, data):
        global wrist_camera_image
        wrist_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

class Top_Camera_Subscriber(Node):
    def __init__(self):
        super().__init__('top_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_top',
            self.camera_callback,
            10)

    def camera_callback(self, data):
        global top_camera_image
        top_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

if __name__ == '__main__':
    rclpy.init(args=None)
    configure_reproducibility_from_env()
    policy = _load_policy_for_evaluation()
    alt_model_env = os.getenv("ALT_MODEL_PATH", "").strip()
    if alt_model_env:
        alt_policy = _load_policy_from_path_str(alt_model_env)
    get_end_effector_pose = Get_End_Effector_Pose()
    joy_publisher = Action_Publisher()
    # Persistent home commander
    robot_home_commander = RobotHomeCommander()
    # Send home a few times at the beginning
    robot_home_commander.send_home(repeat=5)
    wrist_camera_subscriber = Wrist_Camera_Subscriber()
    top_camera_subscriber = Top_Camera_Subscriber()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_end_effector_pose)
    executor.add_node(joy_publisher)
    executor.add_node(robot_home_commander)
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(top_camera_subscriber)
    try:
        while rclpy.ok() and not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for node in [get_end_effector_pose, joy_publisher, robot_home_commander, wrist_camera_subscriber, top_camera_subscriber]:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass
