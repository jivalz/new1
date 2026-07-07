#!/usr/bin/env python3
"""Monitors both cars, counts gate crossings, publishes done flags after N laps."""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32


def quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class LapCounterNode(Node):
    def __init__(self):
        super().__init__('lap_counter')
        self.declare_parameter('max_laps', 10)
        self.declare_parameter('gate_x', 7.50)
        self.declare_parameter('gate_y', 1.25)
        self.declare_parameter('gate_radius', 0.8)
        self.declare_parameter('min_lap_time', 5.0)

        self.max_laps = self.get_parameter('max_laps').value
        self.gate_x = self.get_parameter('gate_x').value
        self.gate_y = self.get_parameter('gate_y').value
        self.gate_r = self.get_parameter('gate_radius').value
        self.min_lap_time = self.get_parameter('min_lap_time').value

        # Lead car state
        self.lead_laps = 0
        self.lead_last_gate_time = self.get_clock().now()
        self.lead_in_gate = None
        self.lead_has_started = False
        self.lead_stopped = False

        # Ego car state
        self.ego_laps = 0
        self.ego_last_gate_time = self.get_clock().now()
        self.ego_in_gate = None
        self.ego_has_started = False
        self.ego_stopped = False

        # Subscriptions
        self.create_subscription(Odometry, '/lead/odom', self.lead_odom_cb, 10)
        self.create_subscription(Odometry, '/ego/odom', self.ego_odom_cb, 10)

        # Done flags only — no cmd_vel publishing
        self.lead_done_pub = self.create_publisher(Bool, '/lead/done', 10)
        self.ego_done_pub = self.create_publisher(Bool, '/ego/done', 10)
        self.current_lap_pub = self.create_publisher(Int32, '/ego/current_lap', 10)

        self.get_logger().info(
            f'Lap counter: max={self.max_laps} laps, '
            f'gate=({self.gate_x:.1f}, {self.gate_y:.1f}) r={self.gate_r}m'
        )

    def _check_gate(self, x, y, in_gate, last_time, has_started):
        dist = math.hypot(x - self.gate_x, y - self.gate_y)
        now = self.get_clock().now()
        
        # Initialization on first tick
        if in_gate is None:
            if dist < self.gate_r:
                # Spawned inside the gate. Race starts when it LEAVES the gate.
                return True, now, False, False
            else:
                # Spawned outside the gate. Race starts when it ENTERS the gate.
                return False, now, False, False

        elapsed = (now - last_time).nanoseconds / 1e9

        if dist < self.gate_r:
            if not in_gate:
                if not has_started:
                    # It was outside, now entering for the first time. The race begins!
                    return True, now, False, True
            return True, last_time, False, has_started
        else:
            if in_gate:
                if not has_started:
                    # It was inside, now leaving for the first time. The race begins!
                    return False, now, False, True
                elif elapsed > self.min_lap_time:
                    # It has already started and is now leaving the gate
                    return False, now, True, True
            return False, last_time, False, has_started

    def lead_odom_cb(self, msg: Odometry):
        if self.lead_stopped:
            self.lead_done_pub.publish(Bool(data=True))
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.lead_in_gate, self.lead_last_gate_time, crossed, self.lead_has_started = self._check_gate(
            x, y, self.lead_in_gate, self.lead_last_gate_time, self.lead_has_started
        )
        if crossed:
            self.lead_laps += 1
            self.get_logger().info(
                f'[LEAD] Lap {self.lead_laps}/{self.max_laps} completed'
            )
            if self.lead_laps >= self.max_laps:
                self.lead_stopped = True
                self.lead_done_pub.publish(Bool(data=True))
                self.get_logger().info('[LEAD] MAX LAPS — DONE')

    def ego_odom_cb(self, msg: Odometry):
        if self.ego_stopped:
            self.ego_done_pub.publish(Bool(data=True))
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ego_in_gate, self.ego_last_gate_time, crossed, self.ego_has_started = self._check_gate(
            x, y, self.ego_in_gate, self.ego_last_gate_time, self.ego_has_started
        )
        if crossed:
            self.ego_laps += 1
            self.current_lap_pub.publish(Int32(data=self.ego_laps))
            self.get_logger().info(
                f'[EGO]  Lap {self.ego_laps}/{self.max_laps} completed'
            )
            if self.ego_laps >= self.max_laps:
                self.ego_stopped = True
                self.ego_done_pub.publish(Bool(data=True))
                self.get_logger().info('[EGO]  MAX LAPS — DONE')


def main(args=None):
    rclpy.init(args=args)
    node = LapCounterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
