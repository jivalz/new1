#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Int32, Bool

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw).
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return yaw_z

class MegaDaggerExpertNode(Node):
    def __init__(self):
        super().__init__('mega_dagger_expert_node')

        # --- Parameters ---
        self.declare_parameter('data_dir', '')
        # Provide any arbitrary number of lanes here. Default lane is the first index (0).
        self.declare_parameter('waypoint_files', ['inner_lane.npz', 'outer_lane.npz'])
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('max_speed', 2.0)
        self.declare_parameter('wheelbase', 0.33)
        
        # Lane Switching thresholds
        self.declare_parameter('opponent_dist_threshold', 3.0)
        self.declare_parameter('opponent_angle_threshold', math.pi / 4.0)
        self.declare_parameter('lane_occupancy_threshold', 0.5)
        self.declare_parameter('cooldown_time', 2.0) # Seconds to wait before switching again
        self.declare_parameter('steering_smooth_factor', 0.1) # 1.0 = instant, lower = smoother
        self.declare_parameter('expert_id', 1)

        # Read parameters
        data_dir = self.get_parameter('data_dir').value
        if not data_dir or not os.path.exists(data_dir):
            import os
            pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(pkg_root, 'data', 'waypoint_files')

        wp_files = self.get_parameter('waypoint_files').value
        self.Ld = self.get_parameter('lookahead_distance').value
        self.max_speed = self.get_parameter('max_speed').value
        self.wheelbase = self.get_parameter('wheelbase').value
        
        self.opp_dist_thresh = self.get_parameter('opponent_dist_threshold').value
        self.opp_angle_thresh = self.get_parameter('opponent_angle_threshold').value
        self.lane_occ_thresh = self.get_parameter('lane_occupancy_threshold').value
        self.cooldown_time = self.get_parameter('cooldown_time').value
        self.steering_smooth_factor = self.get_parameter('steering_smooth_factor').value
        self.expert_id = self.get_parameter('expert_id').value

        import random
        random.seed(self.expert_id)
        np.random.seed(self.expert_id)
        
        from scripts.cbf import CBFdf
        self.cbf = CBFdf(alpha=0.8, gamma=0.5)

        # --- Load Waypoints ---
        import os
        self.lanes = [] 
        for f in wp_files:
            file_path = os.path.join(data_dir, f)
            try:
                data = np.load(file_path)
                if 'waypoints' in data:
                    wps = data['waypoints'][:, :2]
                elif 'poses' in data:
                    wps = data['poses'][:, :2]
                else:
                    k = list(data.keys())[0]
                    wps = data[k][:, :2]
                
                # Apply tail trimming and blending (same as pure_pursuit_expert1_node)
                pts = wps
                dists_to_start = np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])
                half_n = len(dists_to_start) // 2
                if half_n > 50:
                    dists_to_start[:half_n] = 9999.0
                    best_end_idx = np.argmin(dists_to_start)
                    pts = pts[:best_end_idx]
                    
                # Blend the tail
                offset_x = pts[0, 0] - pts[-1, 0]
                offset_y = pts[0, 1] - pts[-1, 1]
                blend_n = min(20, len(pts))
                for i in range(blend_n):
                    weight = (i + 1) / blend_n
                    idx = -blend_n + i
                    pts[idx, 0] += offset_x * weight
                    pts[idx, 1] += offset_y * weight

                self.lanes.append(pts)
                self.get_logger().info(f'Loaded {len(pts)} waypoints from {file_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load {file_path}: {e}')

        if not self.lanes:
            self.get_logger().fatal("No waypoint files successfully loaded. Exiting...")
            return

        self.num_lanes = len(self.lanes)
        self.default_lane_idx = 0
        self.active_lane_idx = self.default_lane_idx
        
        # --- State Tracking ---
        self.ego_pose = None
        self.opp_pose = None
        self.last_switch_time = 0.0
        self.last_angular_z = 0.0
        self.done = False

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, '/ego/cmd_vel', 10)
        self.lane_idx_pub = self.create_publisher(Int32, '/active_lane_idx', 10)
        
        self.path_pubs = []
        for i in range(self.num_lanes):
            self.path_pubs.append(self.create_publisher(Path, f'/lane_path_{i}', 10))

        # --- Subscribers ---
        self.create_subscription(Odometry, '/ego/odom', self.ego_odom_cb, 10)
        self.create_subscription(Odometry, '/opp/odom', self.opp_odom_cb, 10)
        self.create_subscription(Bool, '/ego/done', self.done_cb, 10)

        # Timer to continually publish Paths for RViz
        self.create_timer(1.0, self._publish_path)

    def opp_odom_cb(self, msg):
        self.opp_pose = msg.pose.pose

    def done_cb(self, msg):
        if msg.data:
            self.done = True

    def ego_odom_cb(self, msg):
        """
        Main control loop triggered on ego odometry updates.
        Executes the Pipeline: 
        Opponent Check -> Lane Selector -> Active WP -> Pure Pursuit -> Twist Cmd
        """
        if self.done:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return

        self.ego_pose = msg.pose.pose
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # 1. Lane Occupancy Check
        opp_ahead = False
        opp_on_current_lane = False
        
        if self.opp_pose is not None:
            opp_ahead = self.is_opponent_ahead()
            if opp_ahead:
                # Nearest-neighbor search to see if they are on our exact line
                opp_on_current_lane = self.check_lane_occupancy(self.active_lane_idx)
        
        # 2. Lane Selector (with cooldown hysteresis)
        if (current_time - self.last_switch_time) > self.cooldown_time:
            if opp_on_current_lane:
                self.switch_lane(avoid_lane=self.active_lane_idx)
                self.last_switch_time = current_time
            elif not opp_ahead and self.active_lane_idx != self.default_lane_idx:
                # Safe to return to the default racing line ONLY if we have cleared the opponent
                if self.is_safe_to_return():
                    self.active_lane_idx = self.default_lane_idx
                    self.last_switch_time = current_time

        # Publish active lane index for easy visualization/debugging
        idx_msg = Int32()
        idx_msg.data = self.active_lane_idx
        self.lane_idx_pub.publish(idx_msg)

        # 3. Active Waypoint Set
        active_lane = self.lanes[self.active_lane_idx]
        
        # 4. Pure Pursuit
        target_point = self._find_lookahead(active_lane)
        
        if target_point is not None:
            self._compute_and_publish_twist(target_point)

    # ================= Helper Functions =================

    def distance_to_opponent(self):
        """Calculates Euclidean distance to the opponent vehicle."""
        if self.ego_pose is None or self.opp_pose is None:
            return float('inf')
        dx = self.opp_pose.position.x - self.ego_pose.position.x
        dy = self.opp_pose.position.y - self.ego_pose.position.y
        return math.hypot(dx, dy)

    def is_opponent_ahead(self):
        """
        Checks if the opponent is within the distance threshold and 
        in front of the ego vehicle.
        """
        dist = self.distance_to_opponent()
        if dist > self.opp_dist_thresh:
            return False

        yaw = euler_from_quaternion(
            self.ego_pose.orientation.x, self.ego_pose.orientation.y, 
            self.ego_pose.orientation.z, self.ego_pose.orientation.w
        )
        
        dx = self.opp_pose.position.x - self.ego_pose.position.x
        dy = self.opp_pose.position.y - self.ego_pose.position.y
        
        # Project opponent position onto ego's forward longitudinal axis
        forward_dist = dx * math.cos(yaw) + dy * math.sin(yaw)
        
        # Any opponent with a positive forward distance is "ahead"
        return forward_dist > 0.0

    def is_safe_to_return(self):
        """Ensures the opponent is safely behind before returning to the racing line."""
        if self.opp_pose is None or self.ego_pose is None:
            return True
            
        dist = self.distance_to_opponent()
        if dist > (self.opp_dist_thresh + 1.0):
            return True
            
        yaw = euler_from_quaternion(
            self.ego_pose.orientation.x, self.ego_pose.orientation.y, 
            self.ego_pose.orientation.z, self.ego_pose.orientation.w
        )
        
        dx = self.opp_pose.position.x - self.ego_pose.position.x
        dy = self.opp_pose.position.y - self.ego_pose.position.y
        
        # Project opponent position onto ego's forward longitudinal axis
        forward_dist = dx * math.cos(yaw) + dy * math.sin(yaw)
        
        # Make the safety margin larger to ensure we don't cut back too early at turns
        return forward_dist < -0.9

    def find_closest_point(self, pose, lane):
        """
        Nearest-neighbor search using NumPy. 
        Calculates distance from a given pose to all waypoints in a lane array.
        Returns the index and distance of the closest point.
        """
        px, py = pose.position.x, pose.position.y
        diffs = lane - np.array([px, py])
        dists = np.linalg.norm(diffs, axis=1)
        min_idx = np.argmin(dists)
        return min_idx, dists[min_idx]

    def check_lane_occupancy(self, lane_idx):
        """Checks if the opponent is currently located physically on a specific lane."""
        if self.opp_pose is None:
            return False
            
        lane = self.lanes[lane_idx]
        _, min_dist = self.find_closest_point(self.opp_pose, lane)
        return min_dist < self.lane_occ_thresh

    def switch_lane(self, avoid_lane):
        """
        Iterates over all available lanes to find one that is NOT occupied 
        by the opponent and sets it as the active lane.
        """
        for i in range(self.num_lanes):
            if i != avoid_lane:
                if not self.check_lane_occupancy(i):
                    self.active_lane_idx = i
                    self.get_logger().info(f'Switching to alternate lane {i}')
                    return
        
        self.get_logger().warn('All alternate lanes blocked. Maintaining current lane.')

    # ================= Core Pure Pursuit Functions =================

    def _find_lookahead(self, lane):
        """Finds the lookahead point on the chosen trajectory."""
        closest_idx, _ = self.find_closest_point(self.ego_pose, lane)
        
        lookahead_point = None
        # Search forward for the first point at least Ld away
        for i in range(closest_idx, len(lane)):
            pt = lane[i]
            dist = math.hypot(pt[0] - self.ego_pose.position.x, pt[1] - self.ego_pose.position.y)
            if dist >= self.Ld:
                lookahead_point = pt
                break
                
        # If reached end of array, wrap around (assuming a closed-loop track)
        if lookahead_point is None:
            for i in range(closest_idx):
                pt = lane[i]
                dist = math.hypot(pt[0] - self.ego_pose.position.x, pt[1] - self.ego_pose.position.y)
                if dist >= self.Ld:
                    lookahead_point = pt
                    break
                    
        return lookahead_point

    def _compute_and_publish_twist(self, target_point):
        """Calculates Pure Pursuit curvature and publishes the resulting Twist."""
        yaw = euler_from_quaternion(
            self.ego_pose.orientation.x, self.ego_pose.orientation.y, 
            self.ego_pose.orientation.z, self.ego_pose.orientation.w
        )
        
        dx = target_point[0] - self.ego_pose.position.x
        dy = target_point[1] - self.ego_pose.position.y
        
        # Calculate angle alpha between robot heading and lookahead vector
        alpha = math.atan2(dy, dx) - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        
        # Adjust speed if the opponent is nearby to avoid side nudging
        current_speed = self.max_speed
        dist_to_opp = self.distance_to_opponent()
        
        # Slow down when within a certain distance margin of the lead car and in the same lane
        if dist_to_opp < (self.opp_dist_thresh + 1.0) and self.check_lane_occupancy(self.active_lane_idx):
            current_speed = min(self.max_speed, 2.5)

        # Emergency collision avoidance
        if self.is_opponent_ahead():
            brake_dist = 0.6
            if dist_to_opp < brake_dist:
                # Absolute emergency stop if too close, regardless of lane
                current_speed = 0.0
            elif self.check_lane_occupancy(self.active_lane_idx):
                # Scale speed down smoothly ONLY if opponent is in our lane
                range_dist = max(0.1, self.opp_dist_thresh - brake_dist)
                multiplier = (dist_to_opp - brake_dist) / range_dist
                current_speed = min(current_speed, self.max_speed * multiplier)
            
        # ROS2 Twist translates to V and Omega for standard bases
        # Pure Pursuit Curvature Kappa = 2 * sin(alpha) / Ld
        angular_z = float(2.0 * current_speed * math.sin(alpha) / self.Ld)

        import random
        # Inject MEGA-DAgger Imperfections
        # P(U) = 0.5 per tick is too high for a 20Hz control loop (causes constant jitter and inevitable crashes).
        # We scale it down so the *effective* mistake rate still provides bad data but allows lap completion.
        if random.random() < 0.1:  # 10% chance per tick to make a mistake
            candidates = [
                (current_speed, -angular_z * 0.5), # partial reverse steering (less catastrophic)
                (current_speed, angular_z + np.random.normal(0, 0.3)), # Gaussian steering noise
                (current_speed + np.random.normal(0, 0.2), angular_z), # Gaussian speed noise
                (self.last_linear_x if hasattr(self, 'last_linear_x') else current_speed, 
                 self.last_angular_z if hasattr(self, 'last_angular_z') else angular_z), # delayed action
                (current_speed * 0.5, 0.0) # dropped action (slow down slightly to avoid instant wall crash)
            ]
            
            # Select the imperfection with the highest safety score to avoid immediate trivial crashes
            if self.opp_pose is not None:
                ego_list = [self.ego_pose.position.x, self.ego_pose.position.y, yaw]
                opp_list = [self.opp_pose.position.x, self.opp_pose.position.y]
                
                best_sigma = -float('inf')
                best_cand = candidates[0]
                for v, w in candidates:
                    sig = self.cbf.compute_sigma(ego_list, float(v), float(w), opp_list, dt=0.2)
                    if sig > best_sigma:
                        best_sigma = sig
                        best_cand = (v, w)
                current_speed, angular_z = best_cand
            else:
                current_speed, angular_z = random.choice(candidates)

        self.last_linear_x = float(current_speed)
        self.last_angular_z = float(angular_z)

        twist = Twist()
        twist.linear.x = self.last_linear_x
        twist.angular.z = self.last_angular_z
        
        self.cmd_pub.publish(twist)

    def _publish_path(self):
        """Publishes all loaded lanes as nav_msgs/Path for RViz visualization."""
        for i, lane in enumerate(self.lanes):
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'
            
            for pt in lane:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(pt[0])
                pose.pose.position.y = float(pt[1])
                pose.pose.position.z = 0.0
                path_msg.poses.append(pose)
                
            self.path_pubs[i].publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MegaDaggerExpertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
