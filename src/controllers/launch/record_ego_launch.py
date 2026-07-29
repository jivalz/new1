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
    
    # Ego car spawn (left lane, 1.2m behind)
    # Track goes South (-Y). Left lane is +X (East). Center is -8.0. Left lane center is -6.5.
    ego_x, ego_y, ego_yaw = -6.5, 1.2, -1.57
    
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


        # Joy Node
        TimerAction(
            period=4.0,
            actions=[
                Node(
                    package='joy',
                    executable='joy_node',
                    name='joy_node',
                    output='screen',
                ),
            ],
        ),
        
        # Teleop Twist Joy Node (Driving Ego)
        TimerAction(
            period=4.5,
            actions=[
                Node(
                    package='teleop_twist_joy',
                    executable='teleop_node',
                    name='teleop_twist_joy_node',
                    output='screen',
                    parameters=[{
                        'require_enable_button': False,
                        'axis_linear.x': 1,
                        'axis_angular.yaw': 3,
                        'scale_linear.x': 2.0,
                        'scale_angular.yaw': 2.0,
                    }],
                    remappings=[
                        ('/cmd_vel', '/ego/cmd_vel'),
                    ],
                ),
            ],
        ),
        
        # Data Collector Node (Ego Data)
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package='controllers',
                    executable='data_collector_node',
                    name='data_collector',
                    output='screen',
                    parameters=[{
                        'expert_id': 0,
                        'save_folder': 'ego_data',
                        'workspace': controllers_pkg,
                        'gate_x': -8.0,  # Left lane start
                        'gate_y': 0.0,
                        'gate_yaw': -1.57,
                        'record': True,
                    }],
                    remappings=[
                        ('/scan', '/ego/scan'),
                        ('/odom', '/ego/odom'),
                        ('/cmd_vel', '/ego/cmd_vel'),
                    ],
                ),
            ],
        ),

    ]
    return LaunchDescription(ld)
