#!/usr/bin/env python3
import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.declare_parameter('expert_id',1010)
        self.declare_parameter('data_dir', '') 
        self.declare_parameter('gate_x', 6.22)
        self.declare_parameter('gate_y', 0.98)
        self.declare_parameter('gate_yaw', 2.46) 
        self.declare_parameter('min_lap_time', 5.0)  # seconds
        self.declare_parameter('record', True)
        self.declare_parameter('target_laps', [-1])
        self.declare_parameter('workspace', '')
        self.declare_parameter('save_folder', '')

        self.expert_id = self.get_parameter('expert_id').value
        ws = self.get_parameter('workspace').value
        if not ws:
            ws = os.path.expanduser('~/new1/src/controllers')
        data_dir = os.path.join(ws, 'data')
        
        save_folder = self.get_parameter('save_folder').value
        if save_folder:
            self.data_dir = os.path.join(data_dir, save_folder)
        else:
            self.data_dir = os.path.join(data_dir, f'expert_{self.expert_id}')
            
        os.makedirs(self.data_dir, exist_ok=True)

        gate_x = self.get_parameter('gate_x').value
        gate_y = self.get_parameter('gate_y').value
        gate_yaw = self.get_parameter('gate_yaw').value
        self.min_lap_time = self.get_parameter('min_lap_time').value
        self.recording = self.get_parameter('record').value
        self.target_laps = self.get_parameter('target_laps').value

        self.gate_pos = np.array([gate_x, gate_y])
        self.gate_normal = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/opp/odom', self.opp_odom_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)

        self.latest_scan = None
        self.latest_odom = None
        self.latest_opp_odom = None
        self.latest_cmd = Twist()
        self.expert_active = True  # Always true now that supervisor is removed

        self.prev_gate_sign = None
        self.lap_start_time = None
        self.lap_count = 0
        self.total_laps_saved = self._count_existing_laps()
        self.gate_armed = False  # must travel away before crossing counts
        self.arm_distance = 3.0  # meters from gate before arming

        self.buffer = {
            'timestamps': [],
            'scans': [],
            'poses': [],
            'opp_poses': [],
            'velocities': [],
            'actions': [],
        }

        self.create_timer(0.05, self.record_tick)

        self.get_logger().info(
            f'DataCollector ready — Expert {self.expert_id} | '
            f'Gate=({gate_x:.2f}, {gate_y:.2f}) | '
            f'Saving to {self.data_dir} | '
            f'Existing laps: {self.total_laps_saved}'
        )

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry):
        self.latest_odom = msg
        self._check_gate_crossing(msg)

    def opp_odom_cb(self, msg: Odometry):
        self.latest_opp_odom = msg

    def cmd_cb(self, msg: Twist):
        self.latest_cmd = msg

    def _check_gate_crossing(self, odom: Odometry):
        pos = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
        ])
        d = np.dot(pos - self.gate_pos, self.gate_normal)
        dist_from_gate = np.linalg.norm(pos - self.gate_pos)

        # Arm the gate once the robot is far enough away
        if not self.gate_armed and dist_from_gate > self.arm_distance:
            self.gate_armed = True
            self.get_logger().info(f'Gate armed (robot {dist_from_gate:.1f}m from gate)')

        if self.prev_gate_sign is None:
            self.prev_gate_sign = np.sign(d)
            self.lap_start_time = self.get_clock().now()
            return

        curr_sign = np.sign(d)
        if self.gate_armed and curr_sign != self.prev_gate_sign and curr_sign > 0:
            if dist_from_gate < 3.0:
                now = self.get_clock().now()
                elapsed = (now - self.lap_start_time).nanoseconds / 1e9

                if elapsed > self.min_lap_time:
                    self.lap_count += 1
                    self.get_logger().info(
                        f'🏁 LAP {self.lap_count} complete! '
                        f'Time: {elapsed:.2f}s | '
                        f'Samples in buffer: {len(self.buffer["timestamps"])}'
                    )
                    if self.recording:
                        self._save_lap(elapsed)
                    self._reset_buffer()
                    
                    self.lap_start_time = now
                    self.gate_armed = False  # disarm until robot goes away again

        self.prev_gate_sign = curr_sign

    def record_tick(self):
        if not self.recording or not self.expert_active:
            return
        
        current_lap = self.lap_count + 1
        if -1 not in self.target_laps and current_lap not in self.target_laps:
            return

        if self.latest_scan is None or self.latest_odom is None:
            return

        t = self.get_clock().now().nanoseconds / 1e9
        odom = self.latest_odom

        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        max_r = self.latest_scan.range_max
        ranges = np.where(np.isfinite(ranges), ranges, max_r)

        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        yaw = quat_to_yaw(odom.pose.pose.orientation)
        
        if self.latest_opp_odom is not None:
            opp_px = self.latest_opp_odom.pose.pose.position.x
            opp_py = self.latest_opp_odom.pose.pose.position.y
            opp_yaw = quat_to_yaw(self.latest_opp_odom.pose.pose.orientation)
        else:
            opp_px, opp_py, opp_yaw = 0.0, 0.0, 0.0

        vx = odom.twist.twist.linear.x
        vy = odom.twist.twist.linear.y
        wz = odom.twist.twist.angular.z

        ax = self.latest_cmd.linear.x
        az = self.latest_cmd.angular.z

        self.buffer['timestamps'].append(t)
        self.buffer['scans'].append(ranges)
        self.buffer['poses'].append([px, py, yaw])
        self.buffer['opp_poses'].append([opp_px, opp_py, opp_yaw])
        self.buffer['velocities'].append([vx, vy, wz])
        self.buffer['actions'].append([ax, az])

    def _save_lap(self, lap_time: float):
        if len(self.buffer['timestamps']) < 10:
            self.get_logger().warn('Too few samples — skipping save')
            return

        self.total_laps_saved += 1
        fname = os.path.join(
            self.data_dir, f'lap_{self.total_laps_saved:03d}.npz'
        )
        np.savez_compressed(
            fname,
            timestamps=np.array(self.buffer['timestamps']),
            scans=np.array(self.buffer['scans']),
            poses=np.array(self.buffer['poses']),
            opp_poses=np.array(self.buffer['opp_poses']),
            velocities=np.array(self.buffer['velocities']),
            actions=np.array(self.buffer['actions']),
            lap_time=np.array(lap_time),
            expert_id=np.array(self.expert_id),
        )
        self.get_logger().info(f'💾 Saved {fname} ({lap_time:.2f}s)')

    def _reset_buffer(self):
        for k in self.buffer:
            self.buffer[k] = []

    def _count_existing_laps(self):
        if not os.path.isdir(self.data_dir):
            return 0
        return len([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])

    def destroy_node(self):
        if self.recording and len(self.buffer['timestamps']) > 10:
            self.get_logger().info('Saving partial lap data on shutdown...')
            self._save_lap(0.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
