# Cluster Computing Instructions

The Uni Ulm has access to the BwUniCluster 2.0, a high-performance computer system available to the universities in Baden-Württemberg. 


"The bwUniCluster 2.0 is a massive parallel computer with a total of 848 nodes. Across the entire system, a theoretical peak performance of approx. 1.4 PetaFLOPS and a total memory expansion of approx. 119 TB result.
The base operating system on each node is a Red Hat Enterprise Linux (RHEL) 7.x. The management software for the cluster is KITE, a software environment developed at SCC for the operation of heterogeneous computing clusters."[^1]
[^1]: From https://www.scc.kit.edu/en/services/bwUniCluster_2.0.php


## Registration and Login

See the [Wiki page](https://wiki.bwhpc.de/e/Registration) for detailed instructions on how to register.


## Setting up the phase-field simulation

Login and create a directory named 'Code' in your Home folder. Then navigate to it and download phase-field code from GitLab.

`mkdir Code`\
`cd Code`\
`git clone --branch dev https://gitlab.uni-ulm.de/krill-group/pfsim_ll.git`


## Running phase-field simulations in batch mode

Large jobs (> 8gb mem) must be run in batch mode. An example script (omp_one.sh) is included in the GitLab repository. This should run with no modifications. Assuming you are in '~/Code/pfsim_ll', you can run the following command:

`sbatch -t 4 omp_one.sh ~/Code/pfsim_ll/data/test3d_32cube_sIntvls.h5`

This script script will create a directory 'sim_output' in the user's home folder, and a subfolder with all of the simulation output files will be saved there. It should finish in under 2 minutes. The option `-t 4` sets the max runtime to 4 minutes, so if the job takes longer than 4 min, it will be terminated. This can be easily changed when running the command.

To view the current status of (all) jobs, you can run: `scontrol show job`

Similarly, to run larger simulations (400x400x400 cells), an example script called omp_sim400.sh is available. The only difference to the test script above, is the resources requested. This can be run (from '~/Code/pfsim_ll') as:

`sbatch -t 2500 omp_sim400.sh <path to h5 input file>`

Again, depending on the number of simulation steps, the requested time can easily be modified with the `-t` flag.


## Running phase-field simulations on the log-in node

Sometimes it can be helpful to run a small simulation interactively (e.g., for debugging). This can be done on the login node.

First, navigate to the directory 'pfsim_ll' and create a Python virtual environment.

`python -m venv .venv`\
`source .venv/bin/activate`

Install a few Python modules.

`python -m pip install h5py`\
`python -m pip install natsort`

Load the GNU compiler.

`module load compiler/gnu`

Now you can call Python to run the simulation. Full paths are recommended but relative paths should work as well.

`python run_sim3D.py -f '~/Code/pfsim_ll/data/test3d_32cube_sIntvls.h5'`

The simulation should run, BUT you are running it on a login node. You should only test small simulations here. (See https://wiki.bwhpc.de/e/BwUniCluster2.0/Login#Allowed_Activities_on_Login_Nodes)


## Further reading

General info from the Scientific Computing Center (SCC) about the BwUniCluster
https://www.scc.kit.edu/en/services/bwUniCluster_2.0.php

Workshop with some good instructions for getting started
https://github.com/hpcraink/workshop-parallel-jupyter

Batch System Introduction (from KIT)
https://indico.scc.kit.edu/event/2667/attachments/4974/7529/05_2022-04-07_bwHPC_course_-_intro_batch_system.pdf
