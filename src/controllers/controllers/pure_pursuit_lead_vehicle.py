#!/usr/bin/env python3
"""Pure pursuit expert 2: ego follows pre-recorded waypoints from dataset."""
import os
import glob
import math
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class PurePursuitLeadVehicleNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_lead')
        self.declare_parameter('data_dir', '')
        self.declare_parameter('lookahead', 1.6)
        self.declare_parameter('speed', 3.0)
        self.declare_parameter('max_angular', 2.5)

        data_dir = self.get_parameter('data_dir').value
        if not data_dir:
            data_dir = os.path.expanduser(
                '~/new1/src/controllers/data/lead_data/lead_waypoint'
            )
        self.lookahead = self.get_parameter('lookahead').value
        self.speed = self.get_parameter('speed').value
        self.max_ang = self.get_parameter('max_angular').value

        # Load waypoints from the best recorded lap
        files = sorted(glob.glob(os.path.join(data_dir, 'lap_*.npz')))
        if not files:
            self.get_logger().error(f'No lap files in {data_dir}')
            return

        data = np.load(files[0])
        pts = data['poses'][:, :2]  # (N, 2) → x, y
        
        # 1. Trim backward zigzag tail
        dists_to_start = np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])
        half_n = len(dists_to_start) // 2
        if half_n > 50:
            dists_to_start[:half_n] = 9999.0
            best_end_idx = np.argmin(dists_to_start)
            pts = pts[:best_end_idx]
            
        # 2. Blend the tail so it PERFECTLY meets the start point
        offset_x = pts[0, 0] - pts[-1, 0]
        offset_y = pts[0, 1] - pts[-1, 1]
        blend_n = min(20, len(pts))
        for i in range(blend_n):
            weight = (i + 1) / blend_n
            idx = -blend_n + i
            pts[idx, 0] += offset_x * weight
            pts[idx, 1] += offset_y * weight

        self.waypoints = pts
        self.n_wp = len(self.waypoints)
        self.closest_idx = 0
        self.ego_done = False

        # Subscriptions
        self.create_subscription(Odometry, '/ego/odom', self.ego_odom_cb, 10)
        self.create_subscription(Bool, '/ego/done', self._ego_done_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/ego/cmd_vel', 10)

        # Path visualization
        self.path_pub = self.create_publisher(Path, '/expert2_path', 1)
        self.create_timer(2.0, self._publish_path)

        self.get_logger().info(
            f'Pure pursuit expert 2 initialized. Track points: {len(self.waypoints)}'
        )

    def _publish_path(self):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        for wp in self.waypoints:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = wp[0]
            ps.pose.position.y = wp[1]
            path.poses.append(ps)
        self.path_pub.publish(path)

    def _find_lookahead(self, x, y):
        """Find the lookahead waypoint on the recorded path (wraps around)."""
        dists = np.hypot(self.waypoints[:, 0] - x, self.waypoints[:, 1] - y)
        self.closest_idx = int(np.argmin(dists))

        # Scan forward for first point beyond lookahead distance
        for i in range(self.n_wp):
            idx = (self.closest_idx + i) % self.n_wp
            d = math.hypot(
                self.waypoints[idx, 0] - x,
                self.waypoints[idx, 1] - y,
            )
            if d >= self.lookahead:
                return self.waypoints[idx]

        # Fallback
        idx = (self.closest_idx + self.n_wp // 4) % self.n_wp
        return self.waypoints[idx]

    def _ego_done_cb(self, msg: Bool):
        if msg.data and not self.ego_done:
            self.ego_done = True
            self.cmd_pub.publish(Twist())
            #self.get_logger().info('Expert 2 — laps completed, stopping.')

    def ego_odom_cb(self, msg: Odometry):
        if self.ego_done:
            self.cmd_pub.publish(Twist())
            return

        if not hasattr(self, 'waypoints'):
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quat_to_yaw(msg.pose.pose.orientation)

        goal = self._find_lookahead(x, y)
        gx, gy = float(goal[0]), float(goal[1])

        # Transform goal to robot frame
        dx = gx - x
        dy = gy - y
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

        # Pure pursuit curvature
        ld = math.hypot(local_x, local_y)
        if ld < 0.01:
            ld = 0.01

        curvature = 2.0 * local_y / (ld * ld)

        cmd = Twist()
        cmd.linear.x = self.speed
        cmd.angular.z = max(-self.max_ang, min(self.max_ang,
                                                self.speed * curvature))
        # Slow down on sharp turns
        if cmd.angular.z > 1.57:
            cmd.linear.x = self.speed * (cmd.angular.z / 2.5)
        if cmd.angular.z < -1.57:
            cmd.linear.x = self.speed * (cmd.angular.z / -2.5)
        # Speed up on straights
        if cmd.angular.z < 0.2 and cmd.angular.z > 0:
            cmd.linear.x = self.speed + 0.1
        if cmd.angular.z > -0.2 and cmd.angular.z < 0:
            cmd.linear.x = self.speed + 0.1
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitLeadVehicleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
