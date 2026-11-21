"""openpi's scripts/train.py with our Experiment 1 config registered. Same arguments; run from openpi/.

  python train.py pi05_libero_lora --exp-name clean --data.repo-id exp1/clean [--num-train-steps 2000]
"""
import runpy
import sys

import exp1_config  # noqa: F401  registers pi05_libero_lora

if __name__ == "__main__":  # guarded: the data loader's spawned workers re-import this file
    sys.argv[0] = "scripts/train.py"
    runpy.run_path("scripts/train.py", run_name="__main__")
