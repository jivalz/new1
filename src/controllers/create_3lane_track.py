import os
import math
import numpy as np
import matplotlib.pyplot as plt

class PathBuilder:
    def __init__(self, start_x, start_y, start_yaw):
        self.x = [start_x]
        self.y = [start_y]
        self.yaw = start_yaw
    
    def add_line(self, length, step=0.3):
        n_steps = max(1, int(round(length / step)))
        actual_step = length / n_steps
        for _ in range(n_steps):
            last_x = self.x[-1]
            last_y = self.y[-1]
            self.x.append(last_x + actual_step * math.cos(self.yaw))
            self.y.append(last_y + actual_step * math.sin(self.yaw))
            
    def add_arc(self, radius, angle, step=0.3, turn_left=True):
        arc_length = radius * abs(angle)
        n_steps = max(1, int(round(arc_length / step)))
        d_theta = angle / n_steps if turn_left else -angle / n_steps
        
        center_angle = self.yaw + (math.pi/2.0 if turn_left else -math.pi/2.0)
        cx = self.x[-1] - radius * math.cos(center_angle + math.pi)
        cy = self.y[-1] - radius * math.sin(center_angle + math.pi)
        
        start_angle = center_angle + math.pi
        
        for i in range(1, n_steps + 1):
            theta = start_angle + i * d_theta
            self.x.append(cx + radius * math.cos(theta))
            self.y.append(cy + radius * math.sin(theta))
            
        self.yaw += angle if turn_left else -angle

def main():
    builder = PathBuilder(-8.0, 0.0, -math.pi/2.0)
    
    builder.add_line(6.0)
    builder.add_arc(radius=3.0, angle=math.pi/2.0, turn_left=True)
    builder.add_line(3.0)
    builder.add_arc(radius=3.0, angle=math.pi/2.0, turn_left=True)
    builder.add_line(3.75)
    builder.add_arc(radius=3.0, angle=math.pi, turn_left=False)
    builder.add_line(3.75)
    builder.add_arc(radius=3.0, angle=math.pi/2.0, turn_left=True)
    builder.add_line(3.0)
    builder.add_arc(radius=3.0, angle=math.pi/2.0, turn_left=True)
    builder.add_line(12.0)
    builder.add_arc(radius=3.0, angle=math.pi/2.0, turn_left=True)
    builder.add_line(1.5)
    
    builder.add_arc(radius=4.5, angle=math.radians(35), turn_left=True)
    builder.add_line(2.25)
    builder.add_arc(radius=4.5, angle=math.radians(70), turn_left=False)
    builder.add_line(2.25)
    builder.add_arc(radius=4.5, angle=math.radians(35), turn_left=True)
    
    current_x = builder.x[-1]
    dist_to_go = current_x - (-5.0)
    if dist_to_go > 0:
        builder.add_line(dist_to_go)
        
    builder.add_arc(radius=3.0, angle=math.pi/2.0, turn_left=True)
    
    current_y = builder.y[-1]
    builder.add_line(current_y - 0.0)
    
    x_new = builder.x
    y_new = builder.y
    
    track_width = 4.5
    wall_thickness = 0.1
    wall_height = 0.4
    
    inner_walls = []
    outer_walls = []
    center_lines = []
    
    # Pre-calculate points for inner, outer, and dividers
    pts_center = []
    pts_inner = []
    pts_outer = []
    pts_div1 = []
    pts_div2 = []
    
    for i in range(len(x_new)-1):
        dx = x_new[i+1] - x_new[i]
        dy = y_new[i+1] - y_new[i]
        angle = math.atan2(dy, dx)
        
        ang_inner = angle + math.pi/2.0
        ang_outer = angle - math.pi/2.0
        
        offset = track_width / 2.0
        div_offset = track_width / 6.0
        
        cx = x_new[i]
        cy = y_new[i]
        
        ix = cx + offset * math.cos(ang_inner)
        iy = cy + offset * math.sin(ang_inner)
        
        ox = cx + offset * math.cos(ang_outer)
        oy = cy + offset * math.sin(ang_outer)
        
        l1x = cx + div_offset * math.cos(ang_inner)
        l1y = cy + div_offset * math.sin(ang_inner)
        
        l2x = cx + div_offset * math.cos(ang_outer)
        l2y = cy + div_offset * math.sin(ang_outer)
        
        pts_center.append((cx, cy, angle))
        pts_inner.append((ix, iy))
        pts_outer.append((ox, oy))
        pts_div1.append((l1x, l1y))
        pts_div2.append((l2x, l2y))

    # Add the last point to calculate final distances correctly
    last_dx = x_new[-1] - x_new[-2]
    last_dy = y_new[-1] - y_new[-2]
    last_angle = math.atan2(last_dy, last_dx)
    ang_inner = last_angle + math.pi/2.0
    ang_outer = last_angle - math.pi/2.0
    
    pts_inner.append((x_new[-1] + offset * math.cos(ang_inner), y_new[-1] + offset * math.sin(ang_inner)))
    pts_outer.append((x_new[-1] + offset * math.cos(ang_outer), y_new[-1] + offset * math.sin(ang_outer)))
    pts_div1.append((x_new[-1] + div_offset * math.cos(ang_inner), y_new[-1] + div_offset * math.sin(ang_inner)))
    pts_div2.append((x_new[-1] + div_offset * math.cos(ang_outer), y_new[-1] + div_offset * math.sin(ang_outer)))

    for i in range(len(pts_center)):
        angle = pts_center[i][2]
        
        # Inner wall segment
        idx = pts_inner[i+1][0] - pts_inner[i][0]
        idy = pts_inner[i+1][1] - pts_inner[i][1]
        idist = math.hypot(idx, idy)
        imx = pts_inner[i][0] + idx/2.0
        imy = pts_inner[i][1] + idy/2.0
        inner_walls.append((imx, imy, angle, idist))
        
        # Outer wall segment
        odx = pts_outer[i+1][0] - pts_outer[i][0]
        ody = pts_outer[i+1][1] - pts_outer[i][1]
        odist = math.hypot(odx, ody)
        omx = pts_outer[i][0] + odx/2.0
        omy = pts_outer[i][1] + ody/2.0
        outer_walls.append((omx, omy, angle, odist))
        
        # Divider 1
        d1dx = pts_div1[i+1][0] - pts_div1[i][0]
        d1dy = pts_div1[i+1][1] - pts_div1[i][1]
        d1dist = math.hypot(d1dx, d1dy)
        d1mx = pts_div1[i][0] + d1dx/2.0
        d1my = pts_div1[i][1] + d1dy/2.0
        center_lines.append((d1mx, d1my, angle, d1dist))
        
        # Divider 2
        d2dx = pts_div2[i+1][0] - pts_div2[i][0]
        d2dy = pts_div2[i+1][1] - pts_div2[i][1]
        d2dist = math.hypot(d2dx, d2dy)
        d2mx = pts_div2[i][0] + d2dx/2.0
        d2my = pts_div2[i][1] + d2dy/2.0
        center_lines.append((d2mx, d2my, angle, d2dist))

    # Calculate Optimal Racing Line using rubber-band relaxation
    margin = 0.4  # meters from the wall
    max_offset = (track_width / 2.0) - margin
    
    racing_x = list(x_new)
    racing_y = list(y_new)
    N = len(racing_x)
    
    dist_start_end = math.hypot(racing_x[0] - racing_x[-1], racing_y[0] - racing_y[-1])
    is_closed = (dist_start_end < 0.1)
    
    iters = 2000
    for _ in range(iters):
        for i in range(N):
            if not is_closed and (i == 0 or i == N - 1):
                continue
                
            prev_idx = (i - 1) % N if is_closed else i - 1
            next_idx = (i + 1) % N if is_closed else i + 1
            
            if is_closed and i == N - 1:
                racing_x[N-1] = racing_x[0]
                racing_y[N-1] = racing_y[0]
                continue
                
            nx = (racing_x[prev_idx] + racing_x[next_idx]) / 2.0
            ny = (racing_y[prev_idx] + racing_y[next_idx]) / 2.0
            
            vx = nx - x_new[i]
            vy = ny - y_new[i]
            
            dist = math.hypot(vx, vy)
            if dist > max_offset:
                vx = vx / dist * max_offset
                vy = vy / dist * max_offset
                
            racing_x[i] = x_new[i] + vx
            racing_y[i] = y_new[i] + vy
            
            if is_closed and i == 0:
                racing_x[N-1] = racing_x[0]
                racing_y[N-1] = racing_y[0]
                
    # Generate PNG
    plt.figure(figsize=(10, 10))
    ix_vals = [p[0] for p in pts_inner]
    iy_vals = [p[1] for p in pts_inner]
    ox_vals = [p[0] for p in pts_outer]
    oy_vals = [p[1] for p in pts_outer]
    
    plt.plot(ix_vals, iy_vals, 'b-', label='Inner Wall')
    plt.plot(ox_vals, oy_vals, 'r-', label='Outer Wall')
    plt.plot(x_new, y_new, 'k--', label='Center Line')
    plt.plot(racing_x, racing_y, 'g-', linewidth=2, label='Optimal Racing Line')
    plt.axis('equal')
    plt.legend()
    plt.title("Optimal Racing Line (3-Lane Track)")
    
    png_path = os.path.expanduser('~/new1/src/controllers/data/optimal_racing_line.png')
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close()
    print(f"Saved optimal racing line PNG to {png_path}")
    
    # Save optimal racing line as npz waypoint file for controllers
    wps_path = os.path.expanduser('~/new1/src/controllers/data/ego_data/mpc_trajec/optimal_racing_line_waypoint.npz')
    os.makedirs(os.path.dirname(wps_path), exist_ok=True)
    poses = np.column_stack((racing_x, racing_y))
    np.savez_compressed(wps_path, poses=poses)
    print(f"Saved optimal racing line waypoints to {wps_path}")

    # Generate and save perfect inner, middle, and outer lane waypoints
    lane_offset = 1.5
    inner_x = []
    inner_y = []
    outer_x = []
    outer_y = []
    
    for i in range(len(x_new)):
        if i < len(x_new) - 1:
            dx = x_new[i+1] - x_new[i]
            dy = y_new[i+1] - y_new[i]
        else:
            dx = x_new[i] - x_new[i-1]
            dy = y_new[i] - y_new[i-1]
            
        angle = math.atan2(dy, dx)
        ang_inner = angle + math.pi/2.0
        ang_outer = angle - math.pi/2.0
        
        inner_x.append(x_new[i] + lane_offset * math.cos(ang_inner))
        inner_y.append(y_new[i] + lane_offset * math.sin(ang_inner))
        
        outer_x.append(x_new[i] + lane_offset * math.cos(ang_outer))
        outer_y.append(y_new[i] + lane_offset * math.sin(ang_outer))

    save_dir = os.path.expanduser('~/new1/src/controllers/data/ego_data/3lane_files')
    os.makedirs(save_dir, exist_ok=True)
    
    np.savez_compressed(os.path.join(save_dir, 'middle_lane.npz'), poses=np.column_stack((x_new, y_new)))
    np.savez_compressed(os.path.join(save_dir, 'inner_lane.npz'), poses=np.column_stack((inner_x, inner_y)))
    np.savez_compressed(os.path.join(save_dir, 'outer_lane.npz'), poses=np.column_stack((outer_x, outer_y)))
    print(f"Saved perfect inner, middle, and outer lane waypoints to {save_dir}")

    def make_wall(name, x, y, yaw, length, color="0.8 0.15 0.15 1"):
        return f'''    <model name="{name}"><static>true</static><pose>{x:.4f} {y:.4f} {wall_height/2:.3f} 0 0 {yaw:.4f}</pose><link name="l"><collision name="c"><geometry><box><size>{length:.4f} {wall_thickness} {wall_height}</size></box></geometry></collision><visual name="v"><geometry><box><size>{length:.4f} {wall_thickness} {wall_height}</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual></link></model>'''

    def make_center_line(name, x, y, yaw, length):
        return f'''    <model name="{name}"><static>true</static><pose>{x:.4f} {y:.4f} 0.002 0 0 {yaw:.4f}</pose><link name="l"><visual name="v"><geometry><box><size>{length:.4f} 0.03 0.002</size></box></geometry><material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse></material></visual></link></model>'''

    def generate_world(output_path):
        world_name = os.path.basename(output_path).replace(".world", "")
        world_xml = [
            '<?xml version="1.0"?>',
            '<sdf version="1.6">',
            f'  <world name="{world_name}">',
            '    <include><uri>model://ground_plane</uri></include>',
            '    <include><uri>model://sun</uri></include>',
            '    <scene><shadows>false</shadows></scene>',
            '    <gui fullscreen="0">',
            '      <camera name="user_camera">',
            '        <pose>0 0 25.0 0 1.5708 0</pose>',
            '        <view_controller>orbit</view_controller>',
            '      </camera>',
            '    </gui>',
            '    <physics type="ode">',
            '      <real_time_update_rate>500</real_time_update_rate>',
            '      <max_step_size>0.002</max_step_size>',
            '      <real_time_factor>1</real_time_factor>',
            '    </physics>'
        ]

        for i, (x, y, yaw, length) in enumerate(inner_walls):
            world_xml.append(make_wall(f"iw_{i}", x, y, yaw, length + 0.01, "0.15 0.15 0.8 1"))
            
        for i, (x, y, yaw, length) in enumerate(outer_walls):
            world_xml.append(make_wall(f"ow_{i}", x, y, yaw, length + 0.01, "0.8 0.15 0.15 1"))

        for i, (x, y, yaw, length) in enumerate(center_lines):
            world_xml.append(make_center_line(f"line_{i}", x, y, yaw, length + 0.01))

        start_x = -8.0
        start_y = 0.0
        world_xml.append(f'''    <model name="start_line"><static>true</static>
          <pose>{start_x:.4f} {start_y:.4f} 0.001 0 0 0</pose>
          <link name="l"><visual name="v"><geometry>
            <box><size>{track_width} 0.08 0.001</size></box></geometry>
            <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>
            </material></visual></link>
        </model>''')

        world_xml.append('  </world>')
        world_xml.append('</sdf>')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(world_xml))
        
        print(f"World successfully generated at {output_path}")

    out_path = '/home/rover/new1/src/controllers/worlds/3lane_track.world'
    generate_world(out_path)

if __name__ == '__main__':
    main()
