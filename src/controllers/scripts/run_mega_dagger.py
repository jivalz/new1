#!/usr/bin/env python3
import subprocess
import time
import os
import signal

def main():
    print("======================================")
    print(" MEGA-DAgger Automated Rollout Script")
    print("======================================")
    
    # Clean up any lingering Gazebo processes from previous failed runs before starting
    print("[INFO] Cleaning up lingering Gazebo processes...")
    subprocess.call(['killall', '-9', 'gzserver', 'gzclient'], stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    for expert_id in range(1, 3):
        print(f"\n[INFO] Starting rollout for Expert {expert_id}...")
        
        cmd = ["ros2", "launch", "controllers", "mega_dagger_launch.py", f"expert_id:={expert_id}"]
        process = subprocess.Popen(cmd, preexec_fn=os.setsid)
        
        timeout_seconds = 100
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout_seconds:
                if process.poll() is not None:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user. Shutting down...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            break
            
        if process.poll() is None:
            print(f"[INFO] Rollout timeout reached for Expert {expert_id}. Terminating...")
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait()
            
        print(f"[INFO] Rollout for Expert {expert_id} complete.\n")
        
        # Cleanup lingering Gazebo processes to prevent launch failures
        subprocess.call(['killall', '-9', 'gzserver', 'gzclient'], stderr=subprocess.DEVNULL)
        time.sleep(3)

if __name__ == "__main__":
    main()
