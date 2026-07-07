import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    controllers_pkg = get_package_share_directory('controllers')
    lead_car_pkg = get_package_share_directory('lead_car_description')
    
    # Using the new 3-lane track world
    world = os.path.join(controllers_pkg, 'worlds', '3lane_track.world')
    lead_sdf = os.path.join(lead_car_pkg, 'models', 'lead_car', 'model.sdf')
    
    model_path = ':'.join([
        os.path.join(lead_car_pkg, 'models'),
        os.environ.get('GAZEBO_MODEL_PATH', ''),
    ])
    
    # Lead car spawn (start line)
    lead_x, lead_y, lead_yaw = -8.0, 0.0, -1.57
    
    return LaunchDescription([
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
    ])
