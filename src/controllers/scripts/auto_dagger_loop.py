#!/usr/bin/env python3
import subprocess
import sys
import os
import wandb
import time

def main():
    num_iterations = 5

    print("==================================================")
    print(" MEGA-DAgger Fully Automated Training Loop")
    print(f" Total Iterations: {num_iterations}")
    print("==================================================")

    # Initialize W&B run ID
    run_id = wandb.util.generate_id()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_script = os.path.join(base_dir, 'scripts', 'run_mega_dagger.py')
    train_script = os.path.join(base_dir, 'scripts', 'train_mega_dagger.py')

    for i in range(1, num_iterations + 1):
        print(f"\n\n>>> STARTING DAGGER ITERATION {i}/{num_iterations} <<<")
        print("--------------------------------------------------")
        
        # Step 1: Collect data (run rollout for all 5 experts)
        print(f"\n[Iteration {i}] Phase 1: Data Collection (Rollouts)")
        try:
            subprocess.check_call([sys.executable, run_script])
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Data collection failed during iteration {i}: {e}")
            sys.exit(1)

        # Step 2: Train the novice policy on the newly accumulated data
        print(f"\n[Iteration {i}] Phase 2: Novice Policy Training")
        try:
            # Pass the wandb run id and iteration number so training script can resume the same run
            train_cmd = [
                sys.executable, train_script, 
                '--iteration', str(i),
                '--wandb-id', run_id
            ]
            subprocess.check_call(train_cmd)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Training failed during iteration {i}: {e}")
            sys.exit(1)
            

            
        print(f"\n>>> ITERATION {i} COMPLETE! <<<")

    print("\n==================================================")
    print(" ALL MEGA-DAGGER ITERATIONS FINISHED SUCCESSFULLY!")
    print(" You can now run eval_novice.launch.py to test the final policy.")
    print("==================================================")

if __name__ == '__main__':
    main()
