import time
from pathlib import Path
#import gymnasium as gym
import os
import imageio
import numpy
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
import std_msgs.msg as std_msgs
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
from typing import Literal

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


from pxr import Usd, UsdGeom, Sdf, Gf

import os
os.environ["OMNI_FETCH_ASSETS"] = "1"

bridge = CvBridge()

# Global stop flag for graceful shutdown
stop_event = threading.Event()

tool_pose_xy = [0.0, 0.0] # tool(end effector) pose
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)

from lerobot.policies.act.modeling_act import ACTPolicy

# Select your device
device = "cuda"


# Path resolver to always prefer host-mounted pretrained dir inside Docker
def resolve_pretrained_model_path() -> Path:
    env_path = os.getenv("PRETRAINED_MODEL_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            print(f"\033[36m[RL Finetune][Model Path] Using env PRETRAINED_MODEL_PATH: {p}\033[0m")
            return p
    ws_path = Path("/ws/Scripts/pretrained_model")
    if ws_path.exists():
        print(f"\033[36m[RL Finetune][Model Path] Using host-mounted pretrained dir: {ws_path}\033[0m")
        return ws_path
    host_fallback = Path(os.path.expanduser("~/ur5_nova/Scripts/pretrained_model"))
    if host_fallback.exists():
        print(f"\033[36m[RL Finetune][Model Path] Using fallback local pretrained dir: {host_fallback}\033[0m")
        return host_fallback
    print("\033[33m[RL Finetune][Model Path] Warning: pretrained model dir not found; expected at /ws/Scripts/pretrained_model\033[0m")
    return ws_path

pretrained_policy_path = resolve_pretrained_model_path()
os.environ["PRETRAINED_MODEL_PATH"] = str(pretrained_policy_path)
print(f"\033[36m[RL Finetune] Using pretrained model path: {pretrained_policy_path}\033[0m")

policy_lock = threading.RLock()

def _create_policy_and_optimizer(load_pretrained: bool = True):
    """Create a new policy and optimizer. If load_pretrained False, re-init weights."""
    if load_pretrained:
        p = ACTPolicy.from_pretrained(pretrained_policy_path)
    else:
        p = ACTPolicy.from_pretrained(pretrained_policy_path)
        for module in p.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param.dim() >= 2:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                else:
                    nn.init.zeros_(param)
    p.to(device)
    p.train()
    opt = optim.Adam(filter(lambda par: par.requires_grad, p.parameters()), lr=1e-5)
    return p, opt

policy, optimizer = _create_policy_and_optimizer(load_pretrained=True)

# Finetune hyperparameters
FINETUNE_EVERY_STEPS = 100     # run an update every N steps
FINETUNE_EPOCHS = 2           # how many epochs per update
MINI_BATCH = 48               # batch size per update (reduced to mitigate OOM)
GAMMA = 0.99                  # discount for returns
SAVE_EVERY_UPDATES = 6       # save checkpoint every N finetune calls
MAX_EPISODE_STEPS = 600       # safety cap per episode

updates_done = 0
checkpoint_dir = Path("./rl_finetune").expanduser()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
LATEST_CKPT_NAME = "last_checkpoint.pt"

# Host finetune directory (for mirroring latest checkpoint)
default_host_finetune_dir = \
    "/ws/Scripts/rl_finetune" if os.path.exists("/ws/Scripts") else str(Path("~/ur5_nova/Scripts/rl_finetune").expanduser())
host_finetune_dir = Path(os.getenv("HOST_FINETUNE_DIR", default_host_finetune_dir)).expanduser()
try:
    host_finetune_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Replay buffers (minimal)
rb_states = []      # torch.float32 [2]
rb_images = []      # torch.float32 [3,H,W]
rb_actions = []     # torch.float32 [2]
rb_rewards = []     # float
rb_dones = []       # bool

# Track start of current episode in the buffer
start_buffer_index = 0

def hard_reset_rl(mode: Literal['pretrained','latest','scratch'] = 'pretrained'):
    """Full reset of RL state: replay buffers, model, optimizer, CUDA cache.

    mode options:
      pretrained -> reload original pretrained weights
      latest     -> try to load latest checkpoint (fallback to pretrained)
      scratch    -> random re-init of weights (using kaiming/zeros above)

    Environment variable RL_RESET_MODE can override (latest/scratch/pretrained).
    """
    global policy, optimizer, rb_states, rb_images, rb_actions, rb_rewards, rb_dones, updates_done, start_buffer_index
    env_mode = os.getenv('RL_RESET_MODE', mode).strip().lower()
    if env_mode in ('latest','last'):
        target_mode = 'latest'
    elif env_mode in ('scratch','random','fresh'):
        target_mode = 'scratch'
    else:
        target_mode = 'pretrained'

    with policy_lock:
        # Re-resolve pretrained path each hard reset to pick up host updates
        global pretrained_policy_path
        pretrained_policy_path = resolve_pretrained_model_path()
        os.environ["PRETRAINED_MODEL_PATH"] = str(pretrained_policy_path)
        rb_states.clear(); rb_images.clear(); rb_actions.clear(); rb_rewards.clear(); rb_dones.clear()
        updates_done = 0
        start_buffer_index = 0
        try: del policy
        except NameError: pass
        try: del optimizer
        except NameError: pass
        gc.collect()
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if target_mode == 'latest':
            new_policy, new_opt = _create_policy_and_optimizer(load_pretrained=True)
            loaded = load_latest_checkpoint_if_exists(new_policy, new_opt)
            if not loaded:
                print('\033[33m[hard_reset_rl] No latest checkpoint found; using pretrained.\033[0m')
        elif target_mode == 'scratch':
            new_policy, new_opt = _create_policy_and_optimizer(load_pretrained=False)
        else:
            new_policy, new_opt = _create_policy_and_optimizer(load_pretrained=True)

        # For latest mode if load_latest_checkpoint_if_exists() already loaded weights into global name, rebind
        if target_mode != 'latest':
            policy_ref, opt_ref = new_policy, new_opt
        else:
            # Ensure we have a valid object reference
            policy_ref = new_policy
            opt_ref = new_opt
        # Assign
        globals()['policy'] = policy_ref
        globals()['optimizer'] = opt_ref
        print(f"\033[35m[hard_reset_rl] RL state FULL RESET (mode={target_mode}).\033[0m")
        if target_mode != 'latest':
            try:
                save_checkpoint('startup_after_reset')
            except Exception as e:
                print(f'Reset checkpoint save failed: {e}')

def clear_replay_only():
    global start_buffer_index
    rb_states.clear(); rb_images.clear(); rb_actions.clear(); rb_rewards.clear(); rb_dones.clear()
    start_buffer_index = 0
    print('\033[36m[clear_replay_only] Replay buffer cleared.\033[0m')



def reset_robot():
    node = Node('robot_reset')
    pub = node.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
    
    traj_msg = JointTrajectory()
    traj_msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Home position
    point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point.time_from_start.sec = 3
    
    traj_msg.points = [point]
    pub.publish(traj_msg)
    print("Robot moving to home position...")

# Utility: compute discounted returns and normalize
def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    G = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - float(d))
        returns.append(G)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

def save_checkpoint(tag: str):
    try:
        out = checkpoint_dir / LATEST_CKPT_NAME
        ckpt = {
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tag": tag,
        }
        torch.save(ckpt, out)
        print(f"\033[35mCheckpoint saved: {out} (tag={tag})\033[0m")
        # Mirror to host latest finetune path
        host_out = host_finetune_dir / LATEST_CKPT_NAME
        try:
            torch.save(ckpt, host_out)
            print(f"\033[36m[Mirror] Host latest updated: {host_out}\033[0m")
        except Exception as e:
            print(f"\033[33m[Mirror] Host latest update failed: {e}\033[0m")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

# Helper: load checkpoint by tag (e.g., "startup")
def load_checkpoint_by_tag(tag: str):
    try:
        path = checkpoint_dir / LATEST_CKPT_NAME if tag in ("latest", "last") else checkpoint_dir / f"policy_{tag}.pt"
        data = torch.load(path, map_location=device)
        policy.load_state_dict(data["state_dict"])  # type: ignore[call-arg]
        optimizer.load_state_dict(data.get("optimizer", optimizer.state_dict()))
        policy.to(device)
        policy.train()
        print(f"\033[35mCheckpoint loaded: {path}\033[0m")
        return True
    except Exception as e:
        print(f"Failed to load checkpoint '{tag}': {e}")
        return False

# Helper: find and load the newest checkpoint file
def load_latest_checkpoint_if_exists(policy_obj=None, optimizer_obj=None):
    """Load newest checkpoint into provided policy/optimizer or globals as fallback."""
    pol = policy_obj if policy_obj is not None else globals().get('policy')
    opt = optimizer_obj if optimizer_obj is not None else globals().get('optimizer')
    if pol is None:
        print("No policy object available for loading latest checkpoint.")
        return False
    # Prefer fixed latest file first
    latest_path = checkpoint_dir / LATEST_CKPT_NAME
    if latest_path.exists():
        try:
            data = torch.load(latest_path, map_location=device)
            pol.load_state_dict(data["state_dict"])  # type: ignore[call-arg]
            if opt is not None and "optimizer" in data:
                opt.load_state_dict(data["optimizer"])  # type: ignore[arg-type]
            pol.to(device)
            pol.train()
            print(f"\033[35mResumed from latest checkpoint: {latest_path}\033[0m")
            return True
        except Exception as e:
            print(f"Failed to load fixed latest checkpoint: {e}")
    # Fallback: scan any historical files
    try:
        if not checkpoint_dir.exists():
            return False
        candidates = sorted(checkpoint_dir.glob("policy_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            try:
                data = torch.load(cand, map_location=device)
                pol.load_state_dict(data["state_dict"])  # type: ignore[call-arg]
                if opt is not None and "optimizer" in data:
                    opt.load_state_dict(data["optimizer"])  # type: ignore[arg-type]
                pol.to(device)
                pol.train()
                print(f"\033[35mResumed from checkpoint: {cand}\033[0m")
                return True
            except Exception as e:
                print(f"Failed to load candidate {cand.name}: {e}")
        return False
    except Exception as e:
        print(f"Failed to scan checkpoints: {e}")
        return False

# Lightweight reward-weighted regression update (surrogate for PPO)
# Minimizes MSE(policy(obs) - action) weighted by normalized returns
def finetune_step():
    if len(rb_rewards) < 2:
        return
    # stack tensors
    with policy_lock:
        states = torch.stack(rb_states).to(device)           # [T, 2]
        images = torch.stack(rb_images).to(device)           # [T, 3, H, W]
        actions = torch.stack(rb_actions).to(device)         # [T, 2]
    dones_t = torch.tensor(rb_dones, dtype=torch.float32, device=device)
    returns = compute_returns(rb_rewards, rb_dones, GAMMA)  # [T]

    # build observation dict in batches
    # IMPORTANT: do not pass targets into policy forward to avoid sequence loss path
    # We will compute our own loss between predicted actions and target_actions.

    with torch.enable_grad():
        dataset_size = states.shape[0]
        idx = torch.randperm(dataset_size, device=device)
        for _ in range(FINETUNE_EPOCHS):
            for start in range(0, dataset_size, MINI_BATCH):
                batch_idx = idx[start:start+MINI_BATCH]
                batch_states = states[batch_idx]
                batch_images = images[batch_idx]
                target_actions = actions[batch_idx]          # [B,2]
                adv = returns[batch_idx].unsqueeze(-1)       # [B,1]

                batch_obs = {
                    "observation.state": batch_states,
                    "observation.image": batch_images,
                }

                # Mixed precision safe forward (new API)
                with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                    with policy_lock:
                        out = policy(batch_obs)
                        pred_actions = None
                        if isinstance(out, dict) and "action" in out:
                            pred_actions = out["action"]  # [B,2]

                    if pred_actions is None:
                        # Cannot compute supervised loss if policy does not return actions
                        return

                    # Reward-weighted regression loss (surrogate for PPO objective)
                    loss = ((pred_actions - target_actions)**2 * adv.abs()).mean()

                with policy_lock:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()

    # Clear buffers after an optimization pass (on-policy style)
    rb_states.clear(); rb_images.clear(); rb_actions.clear(); rb_rewards.clear(); rb_dones.clear()

    # Reset start_buffer_index after clear
    global start_buffer_index
    start_buffer_index = 0

    # Save periodic checkpoint
    global updates_done
    updates_done += 1
    if updates_done % SAVE_EVERY_UPDATES == 0:
        save_checkpoint(f"update{updates_done}")

class Get_End_Effector_Pose(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            TFMessage,
            '/isaac_tf',
            self.listener_callback,
            10)
        self.subscription

        self.euler_angles = np.array([0.0, 0.0, 0.0], float)

    def listener_callback(self, data):
        global tool_pose_xy, tbar_pose_xyw

        # 0:tool
        tool_pose = data.transforms[0].transform.translation
        tool_pose_xy[0] = tool_pose.y
        tool_pose_xy[1] = tool_pose.x

        # 1:tbar
        tbar_translation  = data.transforms[1].transform.translation       
        tbar_rotation = data.transforms[1].transform.rotation 
        tbar_pose_xyw[0] = tbar_translation.y
        tbar_pose_xyw[1] = tbar_translation.x
        self.euler_angles[:] = R.from_quat([tbar_rotation.x, tbar_rotation.y, tbar_rotation.z, tbar_rotation.w]).as_euler('xyz', degrees=False)
        tbar_pose_xyw[2] = self.euler_angles[2]

class RewardPublisher(Node):

    def __init__(self):
        super().__init__('rl_reward_publisher')
        self.pub = self.create_publisher(std_msgs.Float32, '/rl/reward', 10)
        self.done_pub = self.create_publisher(std_msgs.Bool, '/rl/done', 10)

    def publish(self, reward: float, done: bool):
        msg = std_msgs.Float32(data=float(reward))
        self.pub.publish(msg)
        self.done_pub.publish(std_msgs.Bool(data=bool(done)))

class Action_Publisher(Node):

    def __init__(self):
        super().__init__('Joy_Publisher')
        # ROS params for runtime tuning
        self.declare_parameter('hz', 10)
        self.declare_parameter('success_threshold', 0.90)
        self.declare_parameter('finetune_every_steps', FINETUNE_EVERY_STEPS)
        self.declare_parameter('max_episode_steps', MAX_EPISODE_STEPS)
        # Stop conditions (save-and-stop)
        self.declare_parameter('accuracy_floor', 0.17)
        self.declare_parameter('peak_min', 0.50)
        self.declare_parameter('drop_threshold', 0.26)
        self.declare_parameter('floor_warmup_steps', 30)
        # Reward bonus for high accuracy
        self.declare_parameter('reward_bonus_threshold', 0.82)
        self.declare_parameter('reward_bonus_value', 0.5)
        # Penalties
        self.declare_parameter('low_acc_penalty_threshold', 0.30)
        self.declare_parameter('low_acc_penalty_value', -0.1)
        self.declare_parameter('drop_stop_penalty_value', -0.5)
        self.declare_parameter('stagnation_penalty_value', -0.5)
        self.declare_parameter('success_reward', 10.0)
        self.declare_parameter('timeout_penalty', -1.0)
        # Action inactivity gating
        self.declare_parameter('no_action_grace_steps', 10)
        self.declare_parameter('action_epsilon', 0.01)

        Hz = int(self.get_parameter('hz').get_parameter_value().integer_value)
        self.success_threshold = float(self.get_parameter('success_threshold').get_parameter_value().double_value or 0.90)
        self.finetune_every_steps = int(self.get_parameter('finetune_every_steps').get_parameter_value().integer_value or FINETUNE_EVERY_STEPS)
        self.max_episode_steps = int(self.get_parameter('max_episode_steps').get_parameter_value().integer_value or MAX_EPISODE_STEPS)
        self.accuracy_floor = float(self.get_parameter('accuracy_floor').get_parameter_value().double_value or 0.21)
        self.peak_min = float(self.get_parameter('peak_min').get_parameter_value().double_value or 0.50)
        self.drop_threshold = float(self.get_parameter('drop_threshold').get_parameter_value().double_value or 0.23)
        self.floor_warmup_steps = int(self.get_parameter('floor_warmup_steps').get_parameter_value().integer_value or 30)
        self.reward_bonus_threshold = float(self.get_parameter('reward_bonus_threshold').get_parameter_value().double_value or 0.85)
        self.reward_bonus_value = float(self.get_parameter('reward_bonus_value').get_parameter_value().double_value or 0.1)
        self.low_acc_penalty_threshold = float(self.get_parameter('low_acc_penalty_threshold').get_parameter_value().double_value or 0.30)
        self.low_acc_penalty_value = float(self.get_parameter('low_acc_penalty_value').get_parameter_value().double_value or 0.1)
        self.drop_stop_penalty_value = float(self.get_parameter('drop_stop_penalty_value').get_parameter_value().double_value or 0.2)
        self.stagnation_penalty_value = float(self.get_parameter('stagnation_penalty_value').get_parameter_value().double_value or 0.5)
        self.success_reward = float(self.get_parameter('success_reward').get_parameter_value().double_value or 10.0)
        self.timeout_penalty = float(self.get_parameter('timeout_penalty').get_parameter_value().double_value or -1.0)
        self.no_action_grace_steps = int(self.get_parameter('no_action_grace_steps').get_parameter_value().integer_value or 40)
        self.action_epsilon = float(self.get_parameter('action_epsilon').get_parameter_value().double_value or 0.01)
        self.no_action_steps = 0
        
        self.pub_joy = self.create_publisher(Joy, '/joy', 10)
        self.joy_commands = Joy()
        self.joy_commands.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.joy_commands.buttons = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.timer = self.create_timer(1/Hz, self.timer_callback)

        # image of a T shape on the table (container-aware with env override)
        images_dir = os.getenv('IMAGES_DIR', 
                               "/ws/images" if os.path.exists("/ws/images") 
                               else os.environ['HOME'] + "/ur5_nova/images")
        self.initial_image = cv2.imread(os.path.join(images_dir, "stand_top_plane.png"))
        if self.initial_image is None:
            raise FileNotFoundError(f"stand_top_plane.png not found in {images_dir}")
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.tool_radius = 10 # millimeters
        self.scale = 1.639344 # mm/pix
        self.C_W = 182 # pix
        self.C_H = 152 # pix
        self.OBL1 = int(150/self.scale)
        self.OBL2 = int(120/self.scale)
        self.OBW = int(30/self.scale)
        # radius of the tool
        self.radius = int(10/self.scale)
        # Overlap masks like in data_collection.py
        self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)
        self.T_image = cv2.imread(os.path.join(images_dir, "stand_top_plane_filled.png"))
        if self.T_image is None:
            raise FileNotFoundError(f"stand_top_plane_filled.png not found in {images_dir}")
        self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
        thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        self.blue_region = cv2.bitwise_not(img_th)
        self.blue_region_sum = cv2.countNonZero(self.blue_region)

        # episode bookkeeping
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_index = 0
        self.best_accuracy = 0.0
        
        # Accuracy stagnation tracking
        self.accuracy_stagnation_steps = 0
        self.last_accuracy = 0.0
        self.accuracy_change_threshold = 0.01  # Minimum change to consider as progress
        self.max_stagnation_steps = 120  # Reset after 120 steps without significant change

        # Initialize start_buffer_index for the first episode
        global start_buffer_index
        start_buffer_index = 0

    def _save_and_stop(self, reason: str):
        tag = f"stop_{reason}_ep{self.episode_index}_step{self.step_count}"
        save_checkpoint(tag)
        print('\033[33m' + f"Stopping episode due to: {reason}" + '\033[0m')
        
        # Robot + sim reset
        reset_robot()
        time.sleep(3.0)
        self.reset_simulation()
        # Hard RL reset (user request: no trace left in memory)
        hard_reset_rl()
        # Reset counters -> new episode can start
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_index += 1
        self.best_accuracy = 0.0
        self.accuracy_stagnation_steps = 0
        self.last_accuracy = 0.0
        start_buffer_index = 0
        print(f"\033[36mStarting new episode (after hard reset): {self.episode_index}\033[0m")


        

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw, start_buffer_index

        self.joy_commands.header.frame_id = "joy"
        self.joy_commands.header.stamp = self.get_clock().now().to_msg()
        
        base_image = copy.copy(self.initial_image)
        # reset region mask
        self.Tbar_region[:] = 0

        x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
        y = int((tool_pose_xy[1]*1000 - 320)/self.scale)
        cv2.circle(base_image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)        
        
        # horizontal part of T
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
        
        # vertical part of T
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

        # Compute accuracy (reward)
        common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
        common_part_sum = cv2.countNonZero(common_part)
        accuracy = common_part_sum/self.blue_region_sum if self.blue_region_sum > 0 else 0.0
        if self.step_count % 10 == 0:
            print(f"step {self.step_count} | ep {self.episode_index} | accuracy(reward): {accuracy:.3f}")

        # Check for accuracy stagnation (new condition)
        accuracy_change = abs(accuracy - self.last_accuracy)
        if accuracy_change < self.accuracy_change_threshold:
            self.accuracy_stagnation_steps += 1
            if self.step_count % 20 == 0 and self.accuracy_stagnation_steps > 0:
                print(f"\033[33m[Stagnation] {self.accuracy_stagnation_steps}/{self.max_stagnation_steps} steps without significant accuracy change\033[0m")
        else:
            if self.accuracy_stagnation_steps > 0:
                print(f"\033[32m[Stagnation] Accuracy changed by {accuracy_change:.4f}, resetting stagnation counter\033[0m")
            self.accuracy_stagnation_steps = 0
        
        self.last_accuracy = accuracy
        
        # Stop condition: accuracy stagnation for too long (apply penalty then reset)
        if self.accuracy_stagnation_steps >= self.max_stagnation_steps:
            penalty = -abs(self.stagnation_penalty_value)
            reward_node.publish(penalty, True)
            self.episode_reward += penalty
            self._save_and_stop(reason=f"stagnation_{self.max_stagnation_steps}_steps")
            return

        # Stop condition 1: accuracy floor breach (immediate) with warmup gate
        if self.step_count >= self.floor_warmup_steps and accuracy < self.accuracy_floor:
            self._save_and_stop(reason=f"floor_{self.accuracy_floor:.2f}")
            return

        # Track best and Stop condition 2: drop from peak after reaching minimum peak
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        if self.best_accuracy >= self.peak_min and (self.best_accuracy - accuracy) >= self.drop_threshold:
            # apply drop-stop penalty before stopping
            penalty = -abs(self.drop_stop_penalty_value)
            reward_node.publish(penalty, True)
            self.episode_reward += penalty
            self._save_and_stop(reason=f"drop_{self.drop_threshold:.2f}_from_{self.best_accuracy:.3f}_to_{accuracy:.3f}")
            return

        # Build observation for policy
        state_t = torch.from_numpy(np.array(tool_pose_xy)).to(torch.float32)
        image_t = torch.from_numpy(base_image).to(torch.float32) / 255
        image_t = image_t.permute(2, 0, 1)
        state_t = state_t.to(device).unsqueeze(0)
        image_t = image_t.to(device).unsqueeze(0)
        obs_t = {
            "observation.state": state_t,
            "observation.image": image_t,
        }

        # Policy action (no inference_mode, we want to capture target for regression)
        with torch.no_grad():
            with policy_lock:
                action_t = policy.select_action(obs_t)  # [1,2]
        numpy_action = action_t.squeeze(0).to("cpu").numpy()
        # Clip actions to safe range [-1, 1]
        numpy_action = np.clip(numpy_action, -1.0, 1.0)

        # --- Reward suppression based on inactivity ---
        is_action = (abs(numpy_action[0]) > self.action_epsilon) or (abs(numpy_action[1]) > self.action_epsilon)
        if is_action:
            if self.no_action_steps >= self.no_action_grace_steps and self.step_count % 10 == 0:
                print("\033[36m[RL] Action detected; reward suppression lifted.\033[0m")
            self.no_action_steps = 0
        else:
            self.no_action_steps += 1
        reward_suppressed = self.no_action_steps >= self.no_action_grace_steps

        # Base reward + bonus + success reward
        reward_value = accuracy + (self.reward_bonus_value if accuracy >= self.reward_bonus_threshold else 0.0)
        reward_value = min(1.0, reward_value)
        if reward_suppressed:
            # Do not give rewards until an action occurs
            if self.step_count % 10 == 0:
                print(f"\033[33m[RL] Reward suppressed (no action for {self.no_action_steps} steps ≥ {self.no_action_grace_steps}).\033[0m")
            reward_value = 0.0

        # Low-accuracy penalty
        if accuracy < self.low_acc_penalty_threshold:
            reward_value -= abs(self.low_acc_penalty_value)

        # Extra bonus for success
        if accuracy >= self.success_threshold:
            reward_value += self.success_reward  # Yeni parametre

        # Publish action
        self.joy_commands.axes[0] = float(numpy_action[0])
        self.joy_commands.axes[1] = float(numpy_action[1])
        self.pub_joy.publish(self.joy_commands)

        # Push transition into buffer
        rb_states.append(state_t.squeeze(0).detach().to("cpu"))
        rb_images.append(image_t.squeeze(0).detach().to("cpu"))
        rb_actions.append(action_t.squeeze(0).detach().to("cpu"))
        print(f"\033[36m[RL] Reward going to model: {reward_value:.4f}\033[0m")
        rb_rewards.append(float(reward_value))
        rb_dones.append(bool(accuracy >= self.success_threshold))

        # Publish reward (bonus/penalty-applied)
        reward_node.publish(reward_value, bool(accuracy >= self.success_threshold))

        self.episode_reward += float(reward_value)
        self.step_count += 1

        # Trigger finetune periodically
        if self.step_count % self.finetune_every_steps == 0:
            try:
                print("\033[36mStarting finetune step...\033[0m")
                finetune_step()
                print("\033[36mFinetune step done.\033[0m")
            except Exception as e:
                print(f"Finetune failed: {e}")

        # Episode termination: success or step cap
        done = (accuracy >= self.success_threshold) or (self.step_count >= self.max_episode_steps)
        if done:
            status = 'SUCCESS' if accuracy >= self.success_threshold else 'TIMEOUT'
            
            if status == 'SUCCESS':
                success_value = 0.0 if reward_suppressed else self.success_reward
                rb_rewards.append(success_value)
                self.episode_reward += success_value
                reward_node.publish(success_value, True)
                print(f"\033[32mSUCCESS! Final reward: {success_value}\033[0m")
                print(f"\033[36m[RL] Final reward going to model: {success_value:.4f}\033[0m")
            
            if status == 'TIMEOUT':
                timeout_value = 0.0 if reward_suppressed else self.timeout_penalty
                rb_rewards.append(timeout_value)
                self.episode_reward += timeout_value
                reward_node.publish(timeout_value, True)
                print(f"\033[31mTIMEOUT! Penalty: {timeout_value}\033[0m")
                print(f"\033[36m[RL] Final reward going to model: {timeout_value:.4f}\033[0m")

            

            # Reset episode counters
            self.episode_reward = 0.0
            self.step_count = 0
            self.episode_index += 1
            self.best_accuracy = 0.0
            self.accuracy_stagnation_steps = 0
            self.last_accuracy = 0.0
            start_buffer_index = len(rb_rewards)

            if status == 'SUCCESS':
                save_checkpoint(f"ep{self.episode_index}_success")

            # Reset robot + simulation
            reset_robot()
            time.sleep(3.0)
            self.reset_simulation()
            # Hard RL reset requested
            hard_reset_rl()
            print(f"\033[36mStarting new episode (hard reset): {self.episode_index}\033[0m")
            return  # next timer tick automatically starts new episode

            

    def reset_simulation(self):
        """
        Sends request to HTTP reset service to reset simulation.
        """
        url = "http://127.0.0.1:8099/reset?reload=1"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("\033[32mReset request queued.\033[0m")
            else:
                print(f"\033[31mReset request failed: {response.status_code} {response.text}\033[0m")
        except Exception as e:
            print(f"\033[31mCould not send reset request: {e}\033[0m")

        

if __name__ == '__main__':
    rclpy.init(args=None)

    # Auto-resume from latest checkpoint unless overridden
    fresh_start_env = os.getenv("RL_FINETUNE_FRESH_START", "0").strip()
    fresh_start = fresh_start_env in ("1", "true", "True")
    if not fresh_start:
        resumed = load_latest_checkpoint_if_exists()
        if not resumed:
            # keep pretrained weights and save startup snapshot
            save_checkpoint("startup")
    else:
        print("Fresh start requested; ignoring existing checkpoints.")
        save_checkpoint("startup")

    get_end_effector_pose = Get_End_Effector_Pose()
    joy_publisher = Action_Publisher()
    reward_node = RewardPublisher()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_end_effector_pose)  
    executor.add_node(joy_publisher)
    executor.add_node(reward_node)

    try:
        while rclpy.ok() and not stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Guard against double-shutdown
        for node in [get_end_effector_pose, joy_publisher, reward_node]:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


