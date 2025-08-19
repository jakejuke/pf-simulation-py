#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=660
#SBATCH --cpus-per-task=40
#SBATCH --mem=10gb
#SBATCH -p single

#Usually you should set
export KMP_AFFINITY=compact,1,0
#export KMP_AFFINITY=verbose,compact,1,0 prints messages concerning the supported affinity
#KMP_AFFINITY Description: https://software.intel.com/en-us/node/524790#KMP_AFFINITY_ENVIRONMENT_VARIABLE

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"

# Load the module
module load compiler/gnu

# Move to code directory and activate Python virtual env
cd /home/ul/ul_funktnan/ul_qgp34/Code/pfsim_ll
source .venv/bin/activate

for ((i=10; i<80; i++))
do
    echo "runing sim validate_$i.h5"
    python run_sim3D.py "/home/ul/ul_funktnan/ul_qgp34/data/cal_sim_input/validate_$i.h5"
done

