#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import numpy as np
import sys
import os
sys.path.insert(0, os.path.expanduser('~/new1/src/controllers'))
from scripts.cbf import CBFdf

class DaggerSupervisorNode(Node):
    def __init__(self):
        super().__init__('dagger_supervisor_node')

        self.declare_parameter('rate', 20.0)
        self.declare_parameter('cbf_alpha', 0.8)
        self.declare_parameter('cbf_gamma', 0.5)
        self.declare_parameter('dt', 0.2)

        alpha = self.get_parameter('cbf_alpha').value
        gamma = self.get_parameter('cbf_gamma').value
        self.dt = self.get_parameter('dt').value

        self.cbf = CBFdf(alpha=alpha, gamma=gamma)
        
        self.last_unsafe_time = self.get_clock().now() - rclpy.time.Duration(seconds=10.0)

        self.latest_scan = None
        self.novice_cmd = Twist()
        self.expert_cmd = Twist()

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Twist, '/novice/cmd_vel', self.novice_cmd_cb, 10)
        self.create_subscription(Twist, '/expert/cmd_vel', self.expert_cmd_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/ego/cmd_vel', 10)
        self.expert_active_pub = self.create_publisher(Bool, '/expert_active', 10)

        rate = self.get_parameter('rate').value
        self.create_timer(1.0 / rate, self.timer_cb)

        self.get_logger().info('DAgger Supervisor initialized.')

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def novice_cmd_cb(self, msg: Twist):
        self.novice_cmd = msg

    def expert_cmd_cb(self, msg: Twist):
        self.expert_cmd = msg

    def timer_cb(self):
        if self.latest_scan is None:
            return

        ranges = list(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        if getattr(self, 'was_expert_active', False):
            ego_speed = self.expert_cmd.linear.x
            ego_omega = self.expert_cmd.angular.z
        else:
            ego_speed = self.novice_cmd.linear.x
            ego_omega = self.novice_cmd.angular.z

        is_safe = self.cbf.is_safe_lidar(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            ego_speed=ego_speed,
            ego_omega=ego_omega,
            dt=self.dt
        )

        current_time = self.get_clock().now()
        
        if not is_safe:
            self.last_unsafe_time = current_time

        # Ensure expert holds control for at least 1.0 second
        time_since_unsafe = (current_time - self.last_unsafe_time).nanoseconds / 1e9
        expert_takeover_active = time_since_unsafe < 1.0

        final_cmd = Twist()
        expert_active_msg = Bool()

        if not expert_takeover_active:
            final_cmd = self.novice_cmd
            expert_active_msg.data = False
            if getattr(self, 'was_expert_active', False):
                self.get_logger().info('✅ Safe state restored. Returning control to Novice.')
        else:
            final_cmd = self.expert_cmd
            expert_active_msg.data = True
            if not getattr(self, 'was_expert_active', False):
                self.get_logger().warn('⚠️ Unsafe state detected! Expert taking control for at least 1s.')
                
        self.was_expert_active = expert_takeover_active

        self.cmd_pub.publish(final_cmd)
        self.expert_active_pub.publish(expert_active_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DaggerSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
