import os
import glob

workspace = '/home/rover/new1/src/controllers'

replacements = [
    ('mega_dagger.', 'scripts.'),
    ('from mega_dagger import', 'from scripts import'),
    ('from scripts.novice_policy import NovicePolicy', 'from scripts.policy_network import PolicyNetwork as NovicePolicy'), 
    ('from mega_dagger.novice_policy import NovicePolicy', 'from scripts.policy_network import PolicyNetwork as NovicePolicy'),
    ('~/ros2_ws/src/mega_dagger', '~/new1/src/controllers'),
    ("package='mega_dagger'", "package='controllers'"),
    ("get_package_share_directory('mega_dagger')", "get_package_share_directory('controllers')"),
    ('ros2 launch mega_dagger mega_dagger.launch.py', 'ros2 launch controllers mega_dagger_launch.py'),
    ('mega_dagger.launch.py', 'mega_dagger_launch.py'),
    ('model = NovicePolicy().to(device)', 'model = NovicePolicy(input_dim=1080, output_dim=2).to(device)'),
    ('self.model = NovicePolicy(input_dim=360, output_dim=2)', 'self.model = NovicePolicy(input_dim=1080, output_dim=2)'),
]

for root, _, files in os.walk(workspace):
    for f in files:
        if not f.endswith('.py'):
            continue
        filepath = os.path.join(root, f)
        if filepath.endswith('fix_imports.py'):
            continue
        with open(filepath, 'r') as file:
            content = file.read()
            
        new_content = content
        for old, new in replacements:
            new_content = new_content.replace(old, new)
            
        if new_content != content:
            with open(filepath, 'w') as file:
                file.write(new_content)
            print(f'Updated {filepath}')
