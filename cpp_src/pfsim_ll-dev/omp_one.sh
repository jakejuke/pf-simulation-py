#!/usr/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=1gb
#SBATCH -p single

#Usually you should set
export KMP_AFFINITY=compact,1,0
#export KMP_AFFINITY=verbose,compact,1,0 prints messages concerning the supported affinity
#KMP_AFFINITY Description: https://software.intel.com/en-us/node/524790#KMP_AFFINITY_ENVIRONMENT_VARIABLE

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"


# # # # # # # My Edits # # # # # # #

if [[ $# -eq 0 ]] ; then
    echo "Usage: $0 input_file.h5 [-c, --code-dir <path>]"
    exit 0
fi

# Mandatory positional arguments
H5_INPUT=$1

# Optionally path to Cpp code can be specified
CODE_DIR=~/Code/pfsim_ll

# Shift out positional argument
shift 1

# Parse flags and optional arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--code-dir) CODE_DIR=$2; echo 'found code'; shift ;;
        -h|--help) 
            echo "Usage: $0 input_file.h5 [-c, --code-dir <path>]"
            exit 0
            ;;
        *)  ;; # default case for no match does nothing
    esac
    shift # move to next input arg
done

# Get name of code dir (the last folder in CODE_DIR)
CODE_DIR="${CODE_DIR%/}" # strip trailing slash (if any)
CODE_BASE="${CODE_DIR##*/}"

# Make tmp dir and copy source code
TEMP_DIR=$(mktemp -d --tmpdir pfsim_XXXXX)
echo "Temp dir created at: $TEMP_DIR"

# Copy C++ source code to temp dir and move there
cp -a $CODE_DIR $TEMP_DIR
cd $TEMP_DIR/$CODE_BASE
echo "Now in: $TEMP_DIR/$CODE_BASE"

# Setup virtual env and install a few packages
python -m venv .venv
source .venv/bin/activate
python -m pip install h5py
python -m pip install natsort

# Load the module for the gcc compiler
module load compiler/gnu

# Run simulation
python run_sim3D.py -f $H5_INPUT -c $TEMP_DIR/$CODE_BASE
