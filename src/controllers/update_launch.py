import os

controllers_launch = '/home/rover/new1/src/controllers/launch/novice_inference_launch.py'
mega_launch = '/home/rover/new1/src/controllers/launch/mega_dagger_launch.py'

with open(controllers_launch, 'r') as f:
    c_lines = f.readlines()

with open(mega_launch, 'r') as f:
    m_lines = f.readlines()

# Extract from controllers launch the imports up to the end of spawn_ego
# Specifically, from `def generate_launch_description():` to the end of `spawn_ego`
c_start = 0
for i, line in enumerate(c_lines):
    if 'def generate_launch_description():' in line:
        c_start = i
        break

c_end = 0
for i, line in enumerate(c_lines):
    if '# Pure Pursuit driving the lead car' in line:
        c_end = i
        break

c_chunk = c_lines[c_start:c_end]

# Now for mega_launch
m_start = 0
for i, line in enumerate(m_lines):
    if 'def generate_launch_description():' in line:
        m_start = i
        break

m_end = 0
for i, line in enumerate(m_lines):
    if '# Lead car policy (Pure Pursuit)' in line:
        m_end = i
        break

# We need to preserve DeclareLaunchArgument('expert_id', ...) inside ld
# So let's insert it manually into the c_chunk

c_chunk_str = "".join(c_chunk)
c_chunk_str = c_chunk_str.replace('ld = [', "ld = [\n        DeclareLaunchArgument('expert_id', default_value='1', description='Expert ID for MEGA-DAgger'),\n")

# Replace the expert_id = LaunchConfiguration('expert_id') at the top of def generate_launch_description():
c_chunk_str = c_chunk_str.replace('def generate_launch_description():\n', "def generate_launch_description():\n    expert_id = LaunchConfiguration('expert_id')\n")

m_chunk_2 = m_lines[m_end:]
# wait, m_chunk_2 uses `return LaunchDescription([...])` but c_chunk uses `ld = [...]` and `return LaunchDescription(ld)`
# in mega_launch, it's just `return LaunchDescription([\n DeclareLaunch... \n # Gazebo...`
# We need to adjust m_chunk_2 so it just appends to `ld`.

# Better yet, let's just write the whole mega_dagger_launch.py

new_mega = """import os
import math
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    expert_id = LaunchConfiguration('expert_id')
    controllers_pkg = get_package_share_directory('controllers')
    lead_car_pkg = get_package_share_directory('lead_car_description')
    ego_car_pkg = get_package_share_directory('ego_car_description')
    
    world = os.path.join(controllers_pkg, 'worlds', '3lane_track.world')
    lead_sdf = os.path.join(lead_car_pkg, 'models', 'lead_car', 'model.sdf')
    ego_sdf = os.path.join(ego_car_pkg, 'models', 'ego_car', 'model.sdf')
    
    model_path = ':'.join([
        os.path.join(lead_car_pkg, 'models'),
        os.path.join(ego_car_pkg, 'models'),
        os.environ.get('GAZEBO_MODEL_PATH', ''),
    ])
    
    # Lead car spawn (start line, center lane)
    lead_x, lead_y, lead_yaw = -8.0, 0.0, -1.57
    
    # Ego car spawn (left lane, 1.2m behind)
    ego_x, ego_y, ego_yaw = -6.5, 1.2, -1.57
    
    ld = [
        DeclareLaunchArgument('expert_id', default_value='1', description='Expert ID for MEGA-DAgger'),

        # Gazebo
        ExecuteProcess(
            cmd=[
                'gazebo', '--verbose', world,
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so',
            ],
            additional_env={'GAZEBO_MODEL_PATH': model_path},
            output='screen',
        ),
        
        # Spawn lead car
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name='spawn_lead',
                    output='screen',
                    arguments=[
                        '-entity', 'lead_car',
                        '-file', lead_sdf,
                        '-x', str(lead_x),
                        '-y', str(lead_y),
                        '-z', '0.01',
                        '-Y', str(lead_yaw),
                    ],
                ),
            ],
        ),
        
        # Spawn ego car
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name='spawn_ego',
                    output='screen',
                    arguments=[
                        '-entity', 'ego_car',
                        '-file', ego_sdf,
                        '-x', str(ego_x),
                        '-y', str(ego_y),
                        '-z', '0.01',
                        '-Y', str(ego_yaw),
                    ],
                ),
            ],
        ),

        # Lead car policy (Pure Pursuit)
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='controllers',
                    executable='pure_pursuit_lead_vehicle',
                    name='lead_policy',
                    output='screen',
                    remappings=[
                        ('/ego/odom', '/lead/odom'),
                        ('/ego/cmd_vel', '/lead/cmd_vel'),
                        ('/ego/done', '/lead/done'),
                    ],
                    parameters=[{
                        'data_dir': os.path.expanduser('~/new1/src/controllers/data/lead_waypoint'),
                        'lookahead': 1.2,
                        'speed': 1.5,
                        'max_angular': 2.5,
                    }],
                ),
            ],
        ),

        # Ego car policy (The Novice being trained)
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='controllers',
                    executable='dagger_inference',
                    name='novice_policy',
                    output='screen',
                    remappings=[
                        ('/scan', '/ego/scan'),
                        ('/cmd_vel', '/novice/cmd_vel'),
                        ('/done', '/ego/done'),
                    ],
                ),
            ],
        ),

        # Lane Switch Expert for ego
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='controllers',
                    executable='mega_dagger_expert',
                    name='expert_policy',
                    output='screen',
                    remappings=[
                        ('/opp/odom', '/lead/odom'),
                        ('/ego/cmd_vel', '/expert/cmd_vel'),
                    ],
                    parameters=[{
                        'expert_id': expert_id,
                        'lookahead_distance': 1.8,
                        'max_speed': 1.7,
                        'opponent_dist_threshold': 1.5,
                        'steering_smooth_factor': 0.05,
                    }],
                ),
            ],
        ),

        # DAgger Supervisor Node
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='controllers',
                    executable='dagger_supervisor',
                    name='dagger_supervisor',
                    output='screen',
                    remappings=[
                        ('/scan', '/ego/scan'),
                    ],
                    parameters=[{
                        'cbf_alpha': 0.8,
                        'cbf_gamma': 0.5,
                        'dt': 0.2,
                        'rate': 20.0,
                    }],
                ),
            ],
        ),

        # Data collector (saves laps every lap for DAgger)
        Node(
            package='controllers',
            executable='dagger_data_collector',
            name='dagger_data_collector',
            output='screen',
            remappings=[
                ('/scan', '/ego/scan'),
                ('/odom', '/ego/odom'),
                ('/cmd_vel', '/ego/cmd_vel'),
                ('/done', '/ego/done'),
            ],
            parameters=[{
                'expert_id': expert_id,
                'gate_x': -8.0,
                'gate_y': 0.0,
                'gate_yaw': -1.57,
                'min_lap_time': 5.0,
                'record': True,
            }],
        ),

        # Lap counter
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='controllers',
                    executable='lap_counter_node',
                    name='lap_counter',
                    output='screen',
                    parameters=[{
                        'max_laps': 5,
                        'gate_x': -8.0,
                        'gate_y': 0.0,
                        'gate_radius': 3.0,
                        'min_lap_time': 5.0,
                    }],
                ),
            ],
        ),
    ]
    return LaunchDescription(ld)
"""

with open(mega_launch, 'w') as f:
    f.write(new_mega)
print("Updated mega_dagger_launch.py")
