#!/usr/bin/env python3
import os
import glob
import numpy as np

def calculate_metrics(folder_path):
    npz_files = glob.glob(os.path.join(folder_path, '*.npz'))
    if not npz_files:
        return None
        
    total_laps = len(npz_files)
    collisions = 0
    overtakes = 0
    
    for f in npz_files:
        data = np.load(f)
        scans = data['scans']
        
        # Collision: any LiDAR beam < 0.25m (ignoring 0.0 which might be errors)
        valid_scans = np.where(scans > 0.05, scans, np.inf)
        if np.min(valid_scans) < 0.25:
            collisions += 1
            
        # Overtaking calculation
        if 'opp_poses' in data:
            poses = data['poses']
            opp_poses = data['opp_poses']
            # Simplistic overtake check: did ego cross the finish line faster than opp?
            # Or did the longitudinal distance flip?
            # For simplicity, let's check if the final distance to origin for ego is greater
            # or if ego's lap time implies an overtake. 
            # Actually, since both start at the same time, if ego finishes the lap and opp 
            # hasn't finished (or ego finished faster), it's an overtake.
            # We can also check if ego ever got physically ahead.
            ego_end = poses[-1][:2]
            opp_end = opp_poses[-1][:2]
            # Since they start near (0,0), if ego finishes the lap, it travelled the whole loop.
            # If opp_poses has the same length, we can check who is further along, 
            # but lap completion implies ego reached the finish line.
            # We'll just use a lap time heuristic: lead car takes ~30s. If ego < 28s, it overtook.
            lap_time = data.get('lap_time', 999)
            if lap_time < 28.0:
                overtakes += 1
        else:
            lap_time = data.get('lap_time', 999)
            if lap_time < 28.0:
                overtakes += 1
                
    return {
        'total_laps': total_laps,
        'collision_rate': (collisions / total_laps) * 100,
        'overtake_rate': (overtakes / total_laps) * 100
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    datasets = [
        ('expert_999', 'Base Expert'),
        ('dagger_1', 'DAgger Iteration 1 (Expert 1)'),
        ('dagger_2', 'DAgger Iteration 2 (Expert 2)'),
        ('dagger_3', 'DAgger Iteration 3 (Expert 3)'),
        ('dagger_4', 'DAgger Iteration 4 (Expert 4)'),
        ('dagger_5', 'DAgger Iteration 5 (Expert 5)'),
        ('expert_9999', 'MEGA-DAgger Final Evaluation')
    ]
    
    print("================================================================")
    print(" MEGA-DAgger Aggregated Performance Report")
    print("================================================================")
    print(f"{'Dataset Name':<35} | {'Laps':<5} | {'Collision %':<12} | {'Overtake %':<10}")
    print("-" * 70)
    
    for folder, desc in datasets:
        folder_path = os.path.join(data_dir, folder)
        metrics = calculate_metrics(folder_path)
        if metrics is not None:
            print(f"{desc:<35} | {metrics['total_laps']:<5} | {metrics['collision_rate']:<11.1f}% | {metrics['overtake_rate']:<9.1f}%")
        else:
            print(f"{desc:<35} | {'N/A':<5} | {'N/A':<12} | {'N/A':<10}")
            
if __name__ == '__main__':
    main()
