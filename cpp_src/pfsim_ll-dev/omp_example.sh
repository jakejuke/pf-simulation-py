#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=1gb
#SBATCH -p single

#Usually you should set
export KMP_AFFINITY=compact,1,0
#export KMP_AFFINITY=verbose,compact,1,0 prints messages concerning the supported affinity
#KMP_AFFINITY Description: https://software.intel.com/en-us/node/524790#KMP_AFFINITY_ENVIRONMENT_VARIABLE

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"

# Set path to code
SCRIPT_DIR=~/Code/pfsim_ll
# Note: on local machine can find directory running the script like this:
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# **But it doesn't work on the cluster with slurm**

# Load the module
module load compiler/gnu

# Move to code directory and activate Python virtual env
cd $SCRIPT_DIR
source .venv/bin/activate

python run_sim3D.py -f $SCRIPT_DIR/data/test3d_32cube_sIntvls.h5
