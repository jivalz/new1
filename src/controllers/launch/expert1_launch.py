import os
import math
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
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
    
    # Ego car spawn (middle lane, 1.5m behind)
    # Track goes South (-Y). Center is -8.0.
    ego_x, ego_y, ego_yaw = -8.0, 1.5, -1.57
    
    ld = [
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

        # Pure Pursuit driving the lead car
        TimerAction(
            period=14.0,
            actions=[
                Node(
                    package='controllers',
                    executable='pure_pursuit_lead_vehicle',
                    name='pure_pursuit_lead',
                    output='screen',
                    remappings=[
                        ('/ego/odom', '/lead/odom'),
                        ('/ego/cmd_vel', '/lead/cmd_vel'),
                        ('/ego/done', '/lead/done'),
                    ],
                ),
            ],
        ),
        
        # Expert driving the ego car
        TimerAction(
            period=14.0,
            actions=[
                Node(
                    package='controllers',
                    executable='mpc_expert1_node',
                    name='mpc_expert1_node',
                    output='screen',
                ),
            ],
        ),
        
        # Lap counter for ego car
        TimerAction(
            period=5.0,
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
