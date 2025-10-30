#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, Image
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import threading
import copy
import time
import cv2
import os
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
from cv_bridge import CvBridge
from math import sin, cos, pi
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pynput import keyboard

bridge = CvBridge()

record_data = False
tool_pose_xy = [0.0, 0.0] # tool(end effector) pose
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 360
vid_W = 640
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
action = np.array([0.0, 0.0], float)


class Get_Poses_Subscriber(Node):

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

class Robot_Keyboard_Controller(Node):

    def __init__(self):
        super().__init__('robot_keyboard_controller')
        
        # Robot control publishers
        # Send Twist commands to MoveIt Servo for real-time control
        self.twist_pub = self.create_publisher(
            TwistStamped, 
            '/servo_node/delta_twist_cmds', 
            10
        )
        
        # Send Joint Trajectory commands for position control (reset to home)
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )
        
        # Keyboard state tracking
        self.keys_pressed = set()
        self.push_time = 0
        self.prev_push_time = 0
        
        # Start keyboard listener in separate thread
        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.listener.start()
        
        # Timer to update robot commands and action values
        self.timer = self.create_timer(0.1, self.update_robot_commands)  # 10Hz control rate (dt = 0.1 s)
        
        print("🤖 Robot Keyboard Control Started!")
        print("Controls:")
        print("  W/S: Move Forward/Backward")
        print("  A/D: Move Left/Right") 
        print("  Q/E: Move Up/Down")
        print("  R: Reset Robot to Home Position")
        print("  Space: Start/Stop Recording")
        print("  ESC: Exit")
        print("  Note: Robot should now move in Isaac Sim!")
        
        # Auto-reset robot to home position when script starts
        self.initial_reset_done = False
        self.create_timer(1.0, self.initial_reset_to_home)

    def on_key_press(self, key):
        global record_data
        
        try:
            # Handle character keys
            if hasattr(key, 'char') and key.char:
                # Special handling for reset key
                if key.char.lower() == 'r':
                    print("\n🔧 R key detected! Calling reset function...")
                    self.reset_robot_to_home()
                    return
                self.keys_pressed.add(key.char.lower())
                
            # Handle special keys
            elif key == keyboard.Key.space:
                self.push_time = time.time()
                dif = self.push_time - self.prev_push_time
                if(dif > 1):  # Debounce - same logic as joystick
                    if(record_data == False):
                        record_data = True
                        print('\033[32m'+'🔴 START RECORDING'+'\033[0m')
                    elif(record_data):
                        record_data = False
                        print('\033[31m'+'⏹️  END RECORDING'+'\033[0m')
                self.prev_push_time = self.push_time
                
            elif key == keyboard.Key.esc:
                print("\n👋 Exiting robot keyboard controller...")
                rclpy.shutdown()
                return False
                
        except AttributeError:
            # Handle special keys that don't have .char
            pass

    def on_key_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.keys_pressed.discard(key.char.lower())
        except AttributeError:
            pass

    def update_robot_commands(self):
        global action
        
        # Create Twist message for robot control
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = "base_link"
        
        # Reset action values
        action[0] = 0.0  # X-axis (left/right)
        action[1] = 0.0  # Y-axis (forward/backward)
        
        # Movement speed for robot (in m/s and rad/s)
        linear_speed = 0.4  # 35cm/s - slightly faster
        vertical_speed = 0.20  # 18cm/s for up/down - slightly faster
        
        # Update robot commands and action based on currently pressed keys
        if 'w' in self.keys_pressed:
            twist_msg.twist.linear.x = linear_speed  # Forward
            action[1] = 1.0
        if 's' in self.keys_pressed:
            twist_msg.twist.linear.x = -linear_speed  # Backward
            action[1] = -1.0
        if 'a' in self.keys_pressed:
            twist_msg.twist.linear.y = linear_speed  # Left
            action[0] = -1.0
        if 'd' in self.keys_pressed:
            twist_msg.twist.linear.y = -linear_speed  # Right
            action[0] = 1.0
        if 'q' in self.keys_pressed:
            twist_msg.twist.linear.z = vertical_speed  # Up
        if 'e' in self.keys_pressed:
            twist_msg.twist.linear.z = -vertical_speed  # Down
            
        # Publish the twist command to move the robot
        self.twist_pub.publish(twist_msg)
        
        # Display current status only when not recording (to avoid cluttering recording output)
        global record_data
        if self.keys_pressed and not record_data:
            keys_str = ', '.join(sorted(self.keys_pressed))
            print(f"\r🎮 Keys: {keys_str:<10} | Action: [{action[0]:+.1f}, {action[1]:+.1f}] | Robot Moving!", end="", flush=True)

    def reset_robot_to_home(self):
        """Reset robot to home position by temporarily restarting arm controller"""
        print("\n🏠 Resetting robot to home position...")
        
        # Clear any pressed keys to stop current servo movements
        self.keys_pressed.clear()
        
        import subprocess
        import time
        
        try:
            print("🔄 Restarting arm controller to bypass servo...")
            
            # Step 1: Deactivate arm controller
            result = subprocess.run([
                'ros2', 'control', 'switch_controllers', 
                '--deactivate', 'arm_controller'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("✅ Arm controller deactivated")
                time.sleep(0.5)  # Wait a bit
                
                # Step 2: Send joint trajectory command
                traj_msg = JointTrajectory()
                traj_msg.header.stamp = self.get_clock().now().to_msg()
                traj_msg.joint_names = [
                    'shoulder_pan_joint',
                    'shoulder_lift_joint', 
                    'elbow_joint',
                    'wrist_1_joint',
                    'wrist_2_joint',
                    'wrist_3_joint'
                ]
                
                # Create trajectory point for home position
                point = JointTrajectoryPoint()
                point.positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Home position for UR5
                point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                point.time_from_start.sec = 3  # Take 3 seconds to reach home
                
                traj_msg.points = [point]
                
                # Step 3: Reactivate arm controller
                result2 = subprocess.run([
                    'ros2', 'control', 'switch_controllers',
                    '--activate', 'arm_controller'
                ], capture_output=True, text=True, timeout=5)
                
                if result2.returncode == 0:
                    print("✅ Arm controller reactivated")
                    time.sleep(0.2)  # Wait a bit more
                    
                    # Step 4: Now send joint trajectory (should work now)
                    for i in range(3):
                        self.joint_trajectory_pub.publish(traj_msg)
                        time.sleep(0.1)
                    
                    print("✅ Robot reset command sent! Moving to home position...")
                    print("⏳ Please wait 3 seconds for movement to complete...")
                else:
                    print("❌ Failed to reactivate arm controller")
            else:
                print("❌ Failed to deactivate arm controller")
                
        except subprocess.TimeoutExpired:
            print("❌ Controller switch timeout")
        except Exception as e:
            print(f"❌ Controller switch error: {e}")

    def initial_reset_to_home(self):
        """Reset robot to home position when script starts (runs once)"""
        if self.initial_reset_done:
            return
            
        print("\n🏠 Initializing robot to home position...")
        
        # Create joint trajectory message
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Create trajectory point for home position
        point = JointTrajectoryPoint()
        point.positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Home position for UR5
        point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point.time_from_start.sec = 2  # Quick reset at startup
        
        traj_msg.points = [point]
        
        # Publish the trajectory
        self.joint_trajectory_pub.publish(traj_msg)
        print("✅ Robot initialized to home position!")
        
        # Mark as done so it doesn't run again
        self.initial_reset_done = True

class Wrist_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_wrist',
            self.camera_callback,
            10)
        self.subscription 

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
        self.subscription 

    def camera_callback(self, data):
        global top_camera_image
        top_camera_image = cv2.resize(bridge.imgmsg_to_cv2(data, "bgr8"), (vid_W, vid_H), cv2.INTER_LINEAR)

class Data_Recorder(Node):

    def __init__(self):
        super().__init__('Data_Recorder')
        self.Hz = 10 # bridge data frequency (dt = 0.1 s)
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.timer = self.create_timer(1/self.Hz, self.timer_callback)
        self.start_recording = False
        self.data_recorded = False

        #### log files for multiple runs are NOT overwritten
        base_dir = os.environ["HOME"] + "/ur5_simulation/src/data_collection/scripts/my_pusht/"
        self.log_dir = base_dir + "data/chunk_000/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        base_vid_dir = base_dir + 'videos/chunk_000/observation.images.'
        self.wrist_vid_dir = base_vid_dir + 'wrist/'
        if not os.path.exists(self.wrist_vid_dir):
            os.makedirs(self.wrist_vid_dir)

        self.top_vid_dir = base_vid_dir + 'top/'
        if not os.path.exists(self.top_vid_dir):
            os.makedirs(self.top_vid_dir)

        self.state_vid_dir = base_vid_dir + 'state/'
        if not os.path.exists(self.state_vid_dir):
            os.makedirs(self.state_vid_dir)

        # image of a T shape on the table
        self.initial_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane.png")
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
        # for reward calculation
        self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)

        # filled image of T shape on the table
        self.T_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane_filled.png")
        self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
        thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        self.blue_region = cv2.bitwise_not(img_th)
        self.blue_region_sum = cv2.countNonZero(self.blue_region)
        
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

        self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index'])
        self.index = 296 # 1 + the last index in the last episode
        self.episode_index = 1 # 1 + the last episode number
        self.frame_index = 0
        self.time_stamp = 0.0
        self.success = False
        self.done = False
        self.column_index = 0
        self.prev_sum = 0.0

        self.wrist_camera_array = []
        self.top_camera_array = []
        self.state_image_array = []

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw, action, wrist_camera_image, top_camera_image, record_data
        
        image = copy.copy(self.initial_image)
        self.Tbar_region[:] = 0

        x = int((tool_pose_xy[0]*1000 + 300)/self.scale)
        y = int((tool_pose_xy[1]*1000 - 320)/self.scale)

        cv2.circle(image, center=(x, y), radius=self.radius, color=(100, 100, 100), thickness=cv2.FILLED)        
        
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
        cv2.fillPoly(image, [pts1_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)
        
        #vertical part of T
        th2 = -tbar_pose_xyw[2] - pi
        dx2 = self.OBL2/2*cos(th2)
        dy2 = self.OBL2/2*sin(th2)
        self.tbar2_ob = [[int(cos(th2)*self.OBL2/2    - sin(th2)*self.OBW/2    + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*self.OBL2/2    - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*self.OBL2/2    + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*(-self.OBW/2)+ dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*(-self.OBW/2)+ dy2 + (1000*y1-320)/self.scale)],
                          [int(cos(th2)*(-self.OBL2/2) - sin(th2)*self.OBW/2   + dx2 + self.C_W + 1000*x1/self.scale), int(sin(th2)*(-self.OBL2/2) + cos(th2)*self.OBW/2   + dy2 + (1000*y1-320)/self.scale)]]  
        pts2_ob = np.array(self.tbar2_ob, np.int32)
        cv2.fillPoly(image, [pts2_ob], (0, 0, 180))
        cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)

        common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
        common_part_sum = cv2.countNonZero(common_part)
        sum = common_part_sum/self.blue_region_sum
        sum_dif = sum - self.prev_sum
        self.prev_sum = sum

        cv2.circle(image, center=(int(self.C_W + 1000*x1/self.scale), int((1000*y1-320)/self.scale)), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)  

        img_msg = bridge.cv2_to_imgmsg(image)  
        self.pub_img.publish(img_msg) 

        if record_data:
            print('\033[32m'+f'RECORDING episode:{self.episode_index}, index:{self.index} sum:{sum}'+'\033[0m')

            if sum >= 0.90:
                self.success = True
                self.done = True
                record_data = False
                print('\033[31m'+'SUCCESS!'+f': {sum}'+'\033[0m')
            else:
                self.success = False

            self.df.loc[self.column_index] = [copy.copy(tool_pose_xy), copy.copy(action), self.episode_index, self.frame_index, self.time_stamp, sum, self.done, self.success, self.index, 0]
            self.column_index += 1
            self.frame_index += 1
            self.time_stamp += 1/self.Hz
            self.index += 1

            self.start_recording = True

            self.wrist_camera_array.append(wrist_camera_image)
            self.top_camera_array.append(top_camera_image)
            self.state_image_array.append(image)

        else:
            if(self.start_recording and self.data_recorded == False):
                print('\033[31m'+'WRITING A PARQUET FILE'+'\033[0m')

                if self.episode_index <= 9:
                    data_file_name = 'episode_00000' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_00000' + str(self.episode_index) + '.mp4'
                elif 9 < self.episode_index <= 99:
                    data_file_name = 'episode_0000' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_0000' + str(self.episode_index) + '.mp4'
                elif 99 < self.episode_index <= 999:
                    data_file_name = 'episode_000' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_000' + str(self.episode_index) + '.mp4'
                elif 999 < self.episode_index <= 9999:
                    data_file_name = 'episode_00' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_00' + str(self.episode_index) + '.mp4'
                else:
                    data_file_name = 'episode_0' + str(self.episode_index) + '.parquet'
                    video_file_name = 'episode_0' + str(self.episode_index) + '.mp4'

                table = pa.Table.from_pandas(self.df)
                pq.write_table(table, self.log_dir + data_file_name)
                print("The parquet file is generated!")

                
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out1 = cv2.VideoWriter(self.wrist_vid_dir + video_file_name, fourcc, self.Hz, (vid_W, vid_H))
                for frame1 in self.wrist_camera_array:
                    out1.write(frame1)
                out1.release()
                print("The wrist video is generated!")
                out2 = cv2.VideoWriter(self.top_vid_dir + video_file_name, fourcc, self.Hz, (vid_W, vid_H))
                for frame2 in self.top_camera_array:
                    out2.write(frame2)
                out2.release()
                print("The top video is generated!")
                out3 = cv2.VideoWriter(self.state_vid_dir + video_file_name, fourcc, self.Hz, (self.initial_image.shape[1], self.initial_image.shape[0]))
                for frame3 in self.state_image_array:
                    out3.write(frame3)
                out3.release()
                print("The state video is generated!")

                # Reset for next episode - exactly like original
                self.data_recorded = True
                self.episode_index += 1  # Increment episode for next recording
                self.frame_index = 0
                self.time_stamp = 0.0
                self.column_index = 0
                self.start_recording = False
                self.success = False
                self.done = False
                
                # Clear arrays for next episode
                self.wrist_camera_array.clear()
                self.top_camera_array.clear()
                self.state_image_array.clear()
                
                # Reset dataframe for next episode
                self.df = self.df.iloc[0:0]  # Clear dataframe
                
                # AUTO RESET ROBOT TO HOME POSITION after episode completion
                self.reset_robot_after_episode()
                
                print(f"🔄 Ready for episode {self.episode_index}")
                self.data_recorded = False

    def reset_robot_after_episode(self):
        """Reset robot to home position after episode completion"""
        print("🤖 Episode completed! Resetting robot to home position...")
        
        # Create joint trajectory message  
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Create trajectory point for home position
        point = JointTrajectoryPoint()
        point.positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]  # Home position for UR5
        point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point.time_from_start.sec = 4  # Take 4 seconds to reach home (a bit slower after episode)
        
        traj_msg.points = [point]
        
        # Publish the trajectory
        joint_trajectory_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        joint_trajectory_pub.publish(traj_msg)
        print("✅ Robot returning to home position for next episode!")


if __name__ == '__main__':
    rclpy.init(args=None)

    get_poses_subscriber = Get_Poses_Subscriber()
    robot_keyboard_controller = Robot_Keyboard_Controller()  # NEW: Robot control!
    wrist_camera_subscriber = Wrist_Camera_Subscriber()
    top_camera_subscriber = Top_Camera_Subscriber()
    data_recorder = Data_Recorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_poses_subscriber)
    executor.add_node(robot_keyboard_controller)  # NEW: Robot control!
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(top_camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = get_poses_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join() 