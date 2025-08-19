#ifndef SRC_MPI_GETTYPE_HPP
#define SRC_MPI_GETTYPE_HPP

#include <src/mesh/mesh2d.hpp>
#include <src/mpi/fundamental.hpp>
#include <mpi.h>

namespace mpi { namespace twd {
	
	/* functions to obtain the type
	 * of a part of a mesh that will be 
	 * transferred
	 */
	
	MPI_Datatype get_col_type(const mesh::twd::Mesh2d& inputmesh,
						std::size_t izero, std::size_t jzero,
						std::size_t m, std::size_t n) {
		
	}
	
	MPI_Datatype get_row_type(const mesh::twd::Mesh2d& inputmesh,
						std::size_t izero, std::size_t jzero,
						std::size_t m, std::size_t n) {
		
	}
	
} /*namespace twd*/ } /*namespace mpi*/ 


#endif
