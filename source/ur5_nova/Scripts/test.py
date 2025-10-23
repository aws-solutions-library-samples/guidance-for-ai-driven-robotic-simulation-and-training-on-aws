#!/usr/bin/env python3
"""
Test Robot Movement - Send simple joint commands to move the robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time


class RobotMover(Node):
    def __init__(self):
        super().__init__('robot_mover')

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(JointState, '/joint_command', 10)

        # Joint names for UR5
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]

        print('🤖 Robot Mover initialized')
        print('📡 Publishing to /joint_command topic')

    def move_to_position(self, positions):
        """Send joint position command"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = positions

        self.joint_pub.publish(msg)
        print(f"📤 Sent positions: {[f'{p:.2f}' for p in positions]}")

    def test_movement(self):
        """Test basic robot movement"""
        print('\n🎯 Starting movement test...')

        # Home position
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        print('🏠 Moving to home position')
        self.move_to_position(home)
        time.sleep(2)

        # Move shoulder
        print('💪 Moving shoulder joint')
        self.move_to_position([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(2)

        # Move elbow
        print('💪 Moving elbow joint')
        self.move_to_position([0.5, -0.5, 0.8, 0.0, 0.0, 0.0])
        time.sleep(2)

        # Return home
        print('🏠 Returning to home')
        self.move_to_position(home)
        time.sleep(2)

        print('✅ Movement test complete!')


def main():
    rclpy.init()

    mover = RobotMover()

    try:
        # Run movement test
        mover.test_movement()

        # Keep node alive
        print('\n🔄 Node running... Press Ctrl+C to stop')
        rclpy.spin(mover)

    except KeyboardInterrupt:
        print('\n🛑 Stopping robot mover')
    finally:
        mover.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




