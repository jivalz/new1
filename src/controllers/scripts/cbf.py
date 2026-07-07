import math
import numpy as np
from typing import Tuple, List

class CBFdf:
    def __init__(self,alpha: float = 0.1, gamma: float = 0.5):
        self.alpha = alpha
        self.gamma = gamma
        self._h_prev: float = None

    def h(self,ego_x: float, ego_y: float, obs_x : float, obs_y : float) -> float:
        hx  = (ego_x - obs_x)**2 + (ego_y - obs_y)**2 - self.alpha**2
        return hx
    
    def sigma(self,ego_x_curr : float,ego_y_curr : float,ego_x_next : float, ego_y_next:float,obs_x : float, obs_y : float) -> float:
        hx_next = self.h(ego_x_next,ego_y_next,obs_x,obs_y)
        hx_curr = self.h(ego_x_curr,ego_y_curr,obs_x,obs_y)
        return hx_next - (1-self.gamma) * hx_curr

    @staticmethod
    def pred_next_pos(x: float, y: float, yaw: float, speed: float, omega: float, dt: float = 0.05) -> Tuple[float, float]:
        # Differential drive kinematic model
        if abs(omega) < 1e-6:
            # Straight-line motion
            x_next = x + speed * math.cos(yaw) * dt
            y_next = y + speed * math.sin(yaw) * dt
        else:
            # Arc motion: R = v / omega
            R = speed / omega
            x_next = x + R * (math.sin(yaw + omega * dt) - math.sin(yaw))
            y_next = y - R * (math.cos(yaw + omega * dt) - math.cos(yaw))
        return x_next, y_next
    
    def compute_sigma(self,ego_pose: List[float],ego_speed: float,ego_omega: float,obs_pose: List[float],dt: float = 0.05) -> float:
        x,y,yaw = ego_pose[0],ego_pose[1],ego_pose[2]
        obsx,obsy = obs_pose[0],obs_pose[1]
        x_next,y_next = self.pred_next_pos(x,y,yaw,ego_speed,ego_omega,dt = dt)
        return self.sigma(x,y,x_next,y_next,obsx,obsy)

    @staticmethod
    def norm(scores : np.ndarray) -> np.ndarray:
        mn,mx = scores.min(),scores.max()
        if mx-mn < 1e-8:
            return np.zeros_like(scores)
        return (scores-mn)/(mx-mn)

    def is_safe(self,ego_pose : List[float],ego_speed: float,ego_omega: float,obs_pose: List[float],dt: float = 0.05) -> bool:
        return self.compute_sigma(ego_pose,ego_speed,ego_omega,obs_pose,dt) >= 0.0

    def is_safe_lidar(self, ranges: List[float], angle_min: float, angle_increment: float, ego_speed: float, ego_omega: float, dt: float = 0.05) -> bool:
        # Local frame prediction (x=0, y=0, yaw=0) — differential drive model
        x_next, y_next = self.pred_next_pos(0.0, 0.0, 0.0, ego_speed, ego_omega, dt=dt)
        
        # We only check points that are relatively close to save computation
        for i, r in enumerate(ranges):
            if r > 3.0 or not math.isfinite(r):
                continue
                
            angle = angle_min + i * angle_increment
            obs_x = r * math.cos(angle)
            obs_y = r * math.sin(angle)
            
            hx_curr = (0 - obs_x)**2 + (0 - obs_y)**2 - self.alpha**2
            hx_next = (x_next - obs_x)**2 + (y_next - obs_y)**2 - self.alpha**2
            
            sigma = hx_next - (1 - self.gamma) * hx_curr
            if sigma < 0.0:
                return False  # Unsafe! Will crash into this wall point.
                
        return True
