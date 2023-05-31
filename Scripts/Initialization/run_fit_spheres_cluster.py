import sys
import os
from pathlib import Path
import argparse

SLURM_COMMAND = """#!/usr/bin/env bash

#SBATCH --exclude=pelkmanslab-slurm-worker-[001-006]
#SBATCH --array=0-{0}%64
#SBATCH --mem=12400
#SBATCH --cpus-per-task=3
#SBATCH -e /data/homes/fcurvaia/outputs/errors_spheres_cluster.txt
#SBATCH -o /data/homes/fcurvaia/outputs/out_spheres_cluster.txt
#SBATCH --time=10:00:00

source ~/.bashrc
conda activate elastix

exec python3.9 /data/homes/fcurvaia/fit_spheres_cluster.py $SLURM_ARRAY_TASK_ID --flds {1} --colnames {2}
"""


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument('--flds', nargs=1, type=str)
    CLI.add_argument('--colnames', nargs="*", type=str, default=[])
    args = CLI.parse_args()

    n = len(list(Path(' '.join(args.flds)).glob('*.h5')))
    # print(SLURM_COMMAND.format(n - 1, ' '.join(args.flds), ' '.join(args.colnames)))
    with open("temp_sph.sh", "w") as f:
        f.write(SLURM_COMMAND.format(n - 1, ' '.join(args.flds), ' '.join(args.colnames)))
    os.system("sbatch temp_sph.sh")
    os.unlink("temp_sph.sh")


if __name__ == "__main__":
    main()
