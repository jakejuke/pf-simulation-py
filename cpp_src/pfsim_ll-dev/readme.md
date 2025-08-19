# Summary
This project provides a header only library to perform phasefield simulations in 2D and 3D for different setups.
Underlying idea is a simple Euler iteration method to solve a differential equation
numerically.
It is completely written in c++, until now compatible with c++07 standard.
Update to a newer standard employing newer features of c++17 is planned for the near future.
To compile and execute change directory to `exec/` and `make simiso2d` to generate
and executable for a simple example. 


# Organization
At the center of the libary stand RAII classes `Mesh2d` and `Mesh3d` that are 
used for memory management. Include these and corresponding utilities 
into a program using for example  
`#include <src/mesh2d.hpp>`  
and create corresponding objects using the constructor. Dimensions need to be specified,
utilities for storing and loading are provided automatically.
For these operations use macros `CHECK_FILE_WRITE` and `CHECK_FILE_READ`, that
can be included from the file `src/aux/checkfileop.hpp`.

For propagation also    
`#include <src/sim2d.hpp>`  
to obtain access to propagation methods. For propagation
two meshes of the same dimension are required, where one
is intended as the current phasefield distribution and the other
serves as destination.
Constants are defined via macros in the file `exec/config2d.hpp`
that needs to be included as well for compilation.

Propagation methods are parallelized using OpenMP. The number of threads
being used can be specified using `export OMP_NUM_THREADS=#` in the current
environment, e.g. bash or zshell. Default is to use the maximum number of
physical theads available (maximal hardware concurrency).

Before running anisotropic simulation or the particle pinning model, generate 
required files by `make misorientation` or `make particlefield2d` executed from the `exec` folder.

Output of simulations is usually stored to `workdir/`, where a new directory has to be created prior to the
simulation run and paths in the main program have to be adjusted accordingly.

Explanations can easily be generalized to the 3d simulation case.

Basic implementation of CUDA features exist but are not yet ready for use.

**Note**: Check for the maximal memory provided by your system.  
Required memory can easily take up several GiB. Estimate for example via:   
DimensionX * DimensionY * LocalBufferSize * sizeof(float) * NumberOfMeshes = 300 * 300 * 32 * 4 * 4 = 46MB  
This is a very rough estimation, up to 10% of additional memory used is to expected.
For a 3d simulation usually a local buffer size of 96 is required, s.t. total memory for 120^3 already is approximately 2.7GB,
and scales cubically. For 300^3 cell expect >40GB of space required.


# Where to find what 
Typically functions are written in C-style for maximal performance and then a wrapper function 
is used to pass on required arguments from the mesh classes. The directory contains:  
- `exec`: contains main programs
- `data`: contains data required, e.g. initial distribution and coefficient maps.
- `src`: Contains source files and collection headers.
	- `src/mesh/`: Header files for mesh classes and utilities concerning such
	- `src/sim/`: Header files from required for simulation. This includes propagation methods as well as functions to generate misorientations etc.
	- `src/aux/`: Some helper classes and functions, usually not required for the main program. Before you start implementing stuff check here if you can use something
	- `src/cuda/`: Contains headers for cuda implementation, RAII storage classes and simulation methods mixed
	- `src/mpi/`: Some unfinished files that would have been useful for MPI implementation
- `test`: test routines used during development, can be useful as templates sometimes


# Contact 
In case of questions refer to <lennart.bosch@uni-ulm.de>. Pls excuse this horribly readme, but I could only spend minimal effort, so do not hesitate to contact me.

# Acknowledgement
Large parts of the implementation are inspired by the lecture library provided in the lecture 
High Performance Computing I by Andreas F. Borchert and Michael Lehn in winter term 2020/2021 
at Ulm University. CUDA features are thereby inspired by an extended version of this library
provided by Andreas F. Borchert in the lecture Parallele Programmierung in C++ in summer term
2021 at Ulm University.

