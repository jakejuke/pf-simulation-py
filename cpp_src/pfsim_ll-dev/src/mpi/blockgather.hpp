#ifndef SRC_MPI_BLOCKSCATTER_HPP
#define SRC_MPI_BLOCKSCATTER_HPP

#include <src/mesh/mesh2d.hpp>
#include <src/mpi/gettype.hpp>
#include <src/aux/slices.hpp>
#include <mpi.h>

namespace mpi { namespace twd {

/* implementation of a function
 * to gather blocks of a parent
 * mesh to the original mesh back
 * the initial mesh, one per each
 * process;
 * input parameters are the "local" 
 * mesh and the original mesh
 * to store the received part in;
 * root process does a simple copy
 * operation
 */


	
} /*namespace twd*/ } /*namespace mpi*/

#endif
