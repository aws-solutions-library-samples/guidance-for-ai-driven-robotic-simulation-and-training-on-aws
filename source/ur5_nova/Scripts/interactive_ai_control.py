#!/usr/bin/env python3
"""
Interactive AI Robot Control - Chat with AI to control robot
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import boto3
import json
import re

class InteractiveRobotController(Node):
    def __init__(self):
        super().__init__('interactive_robot')
        
        self.joint_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        
        print("🤖 Interactive AI Robot Control")
        print("💬 Type commands like: 'reach up', 'move left', 'go to home position'")
        print("🛑 Type 'quit' to exit")
    
    def ask_ai(self, command):
        """Ask AI for movement"""
        prompt = f"""
        You are controlling a UR5 robot arm. Current task: "{command}"
        
        Generate 6 joint angles in radians. Each command should produce DIFFERENT movements:
        - shoulder_pan: -3.14 to 3.14 (base rotation)
        - shoulder_lift: -3.14 to 3.14 (shoulder up/down) 
        - elbow: -3.14 to 3.14 (elbow bend)
        - wrist_1, wrist_2, wrist_3: -3.14 to 3.14 (wrist orientation)
        
        Examples:
        - "reach up": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
        - "move left": [1.57, -1.0, 1.0, 0.0, 0.0, 0.0]
        - "home position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        Respond with ONLY the JSON array for: "{command}"
        """
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}]
        })
        
        response = self.bedrock.invoke_model(
            body=body,
            modelId="anthropic.claude-3-haiku-20240307-v1:0"
        )
        
        result = json.loads(response['body'].read())
        ai_text = result['content'][0]['text']
        
        # Extract positions
        match = re.search(r'\[([-\d\.,\s]+)\]', ai_text)
        if match:
            return [float(x.strip()) for x in match.group(1).split(',')]
        return None
    
    def move_robot(self, positions):
        """Move robot using JointTrajectory"""
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=2, nanosec=0)
        
        msg.points = [point]
        
        self.joint_pub.publish(msg)
        print(f"🤖 Moving: {[f'{p:.2f}' for p in positions]}")
    
    def run_interactive(self):
        """Interactive loop"""
        while True:
            try:
                command = input("\n💬 Command: ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not command:
                    continue
                
                print(f"🧠 AI processing: {command}")
                positions = self.ask_ai(command)
                
                if positions and len(positions) == 6:
                    self.move_robot(positions)
                    print("✅ Movement sent!")
                else:
                    print("❌ AI couldn't generate valid movement")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")

def main():
    rclpy.init()
    
    controller = InteractiveRobotController()
    
    try:
        controller.run_interactive()
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
