import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'controllers'

data_files_list = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
]

for root, dirs, files in os.walk('data'):
    if files:
        install_dir = os.path.join('share', package_name, root)
        file_list = [os.path.join(root, f) for f in files]
        data_files_list.append((install_dir, file_list))

for root, dirs, files in os.walk('weights'):
    if files:
        install_dir = os.path.join('share', package_name, root)
        file_list = [os.path.join(root, f) for f in files]
        data_files_list.append((install_dir, file_list))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files_list,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rover',
    maintainer_email='jivaldhingra@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pure_pursuit_lead_vehicle = controllers.pure_pursuit_lead_vehicle:main',
            'mpc_expert1_node = controllers.mpc_expert1_node:main',
            'lap_counter_node = controllers.lap_counter_node:main',
            'data_collector_node = controllers.data_collector_node:main',
            'teleop = controllers.teleop:main',
            'novice_inference = controllers.novice_inference:main',
            'dagger_inference = controllers.dagger_inference_node:main',
            'mega_dagger_expert = controllers.mega_dagger_expert_node:main',
            'dagger_supervisor = controllers.dagger_supervisor_node:main',
            'dagger_data_collector = controllers.dagger_data_collector_node:main',
            'mpc_expert2_node = controllers.mpc_optimal_trajec_expert2_node:main',
        ],
    },
)
