#ifndef SRC_MESH_SUMSQ2D_HPP
#define SRC_MESH_SUMSQ2D_HPP 1

#include <src/mesh/mesh2d.hpp>
#include <src/mesh/compat2d.hpp>
#include <cstddef>
#include <cassert>

namespace mesh { namespace twd {
	
/* function to determine sum of squares
 * at each lattice site
*/
template <typename T, typename S>
void sumsq(std::size_t m, std::size_t n,
			S* noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig,
			T* origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin,
			T* dest,
			std::ptrdiff_t incRowDest, std::ptrdiff_t incColDest
		) {
	// orig and origin denotes the original mesh data
	// highest value and corresponding order parameter
	// id are stored at places denoted by dest
	
	// loop over each lattice site
	#pragma omp parallel for
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
	
			T sumsq = 0;
			
			// loop all the values at each lattice site
			// loop size is determined by the value of
			// nonzero order parameters
			for (std::size_t u=0; u<noporig[i*incRowNoporig+j*incColNoporig]; ++u) {
				sumsq += origin[i*incRowOrigin+j*incColOrigin+u]
						* origin[i*incRowOrigin+j*incColOrigin+u];
			}
			
			dest[i*incRowDest+j*incColDest] = sumsq;
		}
	}
}

/* wrapper function for C-style implementation
 * to call using mesh objects
*/
template <typename T, typename S>
void sumsq(mesh::twd::Mesh2d<S> &noporig, mesh::twd::Mesh2d<T> &origin, mesh::twd::Mesh2d<T> &dest) {
	// argument origin contains order parameter values
	// argument idmesh contains the corresponding ids
	// argument buf will have stored to id and the 
	// corresponding value
	
	// first check if dest and origin have the same
	// size
	assert(mesh::twd::check_compat(origin, dest));
	
	// call C-style implementation
	sumsq(noporig.xsize(), noporig.ysize(),
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(),
		   dest.ptr(),
		   dest.xinc(), dest.yinc());
}

	

} /*namespace twd*/ } /*namespace mesh*/

#endif //SRC_MESH_SUMSQ2D_HPP
