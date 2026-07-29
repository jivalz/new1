#!/usr/bin/env python3
"""
Basic Spatial Tracking MPC for autonomous racing.
Uses a time-based reference trajectory generated dynamically based on track curvature.
"""
import os
import glob
import math
import numpy as np
from scipy.interpolate import splprep, splev, CubicSpline
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Empty


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class ReferencePath:
    def __init__(self, waypoints_xy: np.ndarray):
        diffs = np.diff(waypoints_xy, axis=0)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        keep = np.concatenate(([True], dists > 0.02))
        pts = waypoints_xy[keep]

        # Fix backward zigzag by trimming the overlapping tail
        dists_to_start = np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])
        half_n = len(dists_to_start) // 2
        if half_n > 50:
            dists_to_start[:half_n] = 9999.0
            best_end_idx = np.argmin(dists_to_start)
            pts = pts[:best_end_idx]

        # Blend the tail so it PERFECTLY meets the start point without any lateral jumps.
        # This prevents the periodic spline from violently whipping to connect them.
        offset_x = pts[0, 0] - pts[-1, 0]
        offset_y = pts[0, 1] - pts[-1, 1]
        blend_n = min(20, len(pts))
        for i in range(blend_n):
            weight = (i + 1) / blend_n  # Gradually goes from 0.05 to 1.0 at the last point
            idx = -blend_n + i
            pts[idx, 0] += offset_x * weight
            pts[idx, 1] += offset_y * weight

        # Fit a perfectly smooth PERIODIC spline to the blended points
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=3.0, per=True)
        u_fine = np.linspace(0, 1, 1000)
        x_fine, y_fine = splev(u_fine, tck)

        # Force perfectly closed loop for the periodic evaluator
        x_fine[-1] = x_fine[0]
        y_fine[-1] = y_fine[0]

        ds = np.hypot(np.diff(x_fine), np.diff(y_fine))
        self.s = np.concatenate(([0.0], np.cumsum(ds)))
        self.total_length = self.s[-1]

        self.cx = CubicSpline(self.s, x_fine, bc_type='periodic')
        self.cy = CubicSpline(self.s, y_fine, bc_type='periodic')

    def eval(self, s):
        s = s % self.total_length
        return float(self.cx(s)), float(self.cy(s))

    def eval_d(self, s):
        s = s % self.total_length
        return float(self.cx(s, 1)), float(self.cy(s, 1))

    def curvature(self, s):
        s = s % self.total_length
        dx, dy = self.eval_d(s)
        ddx, ddy = float(self.cx(s, 2)), float(self.cy(s, 2))
        denom = (dx**2 + dy**2)**1.5
        if denom < 1e-6: return 0.0
        return abs(dx * ddy - dy * ddx) / denom

    def heading(self, s):
        dx, dy = self.eval_d(s)
        return math.atan2(dy, dx)

    def find_closest_s(self, x, y, s_guess, window=2.0, n_samples=40):
        ss = np.linspace(s_guess - window, s_guess + window, n_samples)
        ss = ss % self.total_length
        xs = self.cx(ss)
        ys = self.cy(ss)
        dists = (xs - x)**2 + (ys - y)**2
        return float(ss[np.argmin(dists)])


class BasicMPCSolver:
    def __init__(self, ref_path: ReferencePath, N=10, dt=0.1):
        self.ref = ref_path
        self.N = N
        self.dt = dt
        self.v_max = 3.0        # Backed down from 5.0 to prevent Gazebo physics teleportation/strafing
        self.w_max = 2.5        # Backed down to prevent wheel slip glitches
        self.a_lat_max = 3.5    # Lowered so Gazebo ODE solver doesn't blow up
        self.u_prev = None

    def get_reference_trajectory(self, s0):
        # Generate N dynamically feasible reference points
        refs = []
        s = s0
        for _ in range(self.N):
            k = self.ref.curvature(s)
            # Velocity profile based on curvature
            v_ref = min(self.v_max, math.sqrt(self.a_lat_max / max(k, 1e-3)))
            x, y = self.ref.eval(s)
            th = self.ref.heading(s)
            refs.append((x, y, th, v_ref))
            s += v_ref * self.dt
        return refs

    def solve(self, state0):
        # state0 = [x, y, th, v, w, s]
        x0, y0, th0 = state0[0], state0[1], state0[2]
        s0 = state0[5]
        
        refs = self.get_reference_trajectory(s0)
        n_vars = self.N * 2  # [v, w] per step
        
        if self.u_prev is not None:
            u0 = np.roll(self.u_prev.reshape(self.N, 2), -1, axis=0).flatten()
            u0[-2:] = u0[-4:-2]
        else:
            u0 = np.zeros(n_vars)
            for i in range(self.N):
                u0[i*2] = refs[i][3] # initial guess is v_ref
                
        bounds = []
        for _ in range(self.N):
            bounds.append((0.0, self.v_max))
            bounds.append((-self.w_max, self.w_max))
            
        def cost_fn(u_flat):
            u_all = u_flat.reshape(self.N, 2)
            J = 0.0
            x, y, th = x0, y0, th0
            
            for i in range(self.N):
                v, w = u_all[i]
                
                # Kinematic Euler forward simulation
                x += v * math.cos(th) * self.dt
                y += v * math.sin(th) * self.dt
                th += w * self.dt
                th = wrap_angle(th)
                
                xr, yr, thr, vr = refs[i]
                
                # Tracking cost (Priority 1: Do not hit the wall!)
                J += 85.0 * ((x - xr)**2 + (y - yr)**2)
                
                # Heading cost (Increased to dampen oscillations - forces the bot to align its nose)
                dth = wrap_angle(th - thr)
                J += 20.0 * dth**2
                
                # Speed tracking cost (Keep it fast!)
                J += 500.0 * (v - vr)**2
                
                # Steering magnitude penalty
                J += 1.0 * w**2
                
                # Smoothness cost (Massively increase dw penalty to completely kill fishtailing/jerking)
                if i > 0:
                    dv = v - u_all[i-1, 0]
                    dw = w - u_all[i-1, 1]
                    J += 2.0 * dv**2 + 30.0 * dw**2
            return J

        res = minimize(cost_fn, u0, method='SLSQP', bounds=bounds, options={'maxiter': 10, 'disp': False})
        self.u_prev = res.x
        
        v_opt, w_opt = res.x[0], res.x[1]
        return float(v_opt), float(w_opt)


class BasicMPCNode(Node):
    def __init__(self):
        super().__init__('expert1')

        self.declare_parameter('data_dir', '')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('horizon', 10)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_max', 3.5)

        data_dir = self.get_parameter('data_dir').value
        if not data_dir or not os.path.exists(data_dir):
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(pkg_root, 'data', 'ego_data', 'mpc_trajec')
            
        rate = self.get_parameter('rate').value
        N = self.get_parameter('horizon').value
        dt = self.get_parameter('dt').value

        files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if not files:
            self.get_logger().error(f'No lap files in {data_dir}')
            return

        all_poses = []
        for f in files[:1]:
            d = np.load(f)
            all_poses.append(d['poses'][:, :2])
            self.get_logger().info(f'Loaded {os.path.basename(f)}')

        waypoints = np.vstack(all_poses)
        self.ref_path = ReferencePath(waypoints)
        self.get_logger().info(f'Reference path: length={self.ref_path.total_length:.1f}m')

        self.solver = BasicMPCSolver(self.ref_path, N=N, dt=dt)
        self.solver.v_max = self.get_parameter('v_max').value

        self.ego_state = None
        self.s_progress = 0.0
        self.ego_done = False

        self.create_subscription(Odometry, '/ego/odom', self._ego_odom_cb, 10)
        self.create_subscription(Bool, '/ego/done', self._done_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/ego/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/mpcc/ref_path', 1)

        self.create_service(Empty, '/expert2/reset', self._reset_cb)

        self.create_timer(1.0 / rate, self._control_tick)
        self.create_timer(5.0, self._publish_ref_path)
        self._publish_ref_path()

    def _ego_odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q)
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.ego_state = np.array([p.x, p.y, yaw, vx, wz])

    def _done_cb(self, msg: Bool):
        if msg.data and not self.ego_done:
            self.ego_done = True
            self.cmd_pub.publish(Twist())
            self.get_logger().info('MPC — laps done, stopping.')

    def _reset_cb(self, request, response):
        self._s_initialized = False
        self.solver.u_prev = None
        self.get_logger().info('MPC state reset triggered via service.')
        return response

    def _control_tick(self):
        if self.ego_done or self.ego_state is None:
            return

        x, y, th, v, w = self.ego_state

        if not hasattr(self, '_s_initialized') or not self._s_initialized:
            self.s_progress = self.ref_path.find_closest_s(x, y, 0.0, window=self.ref_path.total_length / 2.0, n_samples=200)
            self._s_initialized = True
        else:
            self.s_progress = self.ref_path.find_closest_s(x, y, self.s_progress, window=3.0)

        state = np.array([x, y, th, v, w, self.s_progress])
        v_cmd, w_cmd = self.solver.solve(state)

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

    def _publish_ref_path(self):
        if not hasattr(self, 'ref_path'):
            return
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        n_pts = 200
        for i in range(n_pts + 1):
            s = self.ref_path.total_length * i / n_pts
            px, py = self.ref_path.eval(s)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = px
            ps.pose.position.y = py
            path.poses.append(ps)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = BasicMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
