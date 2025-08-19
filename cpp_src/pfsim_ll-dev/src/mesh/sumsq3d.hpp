#ifndef SRC_MESH_SUMSQ3D_HPP
#define SRC_MESH_SUMSQ3D_HPP 1

#include <src/mesh/mesh3d.hpp>
#include <src/mesh/compat3d.hpp>
#include <cstddef>
#include <cassert>

namespace mesh { namespace thd {
	
/* function to extract value of maximal
 * order parameter at each lattice site
 * and its corresponding id
*/
template <typename T, typename S>
void sumsq(std::size_t m, std::size_t n, std::size_t k,
			S* noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig, std::ptrdiff_t incLayNoporig,
			T* origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin, std::ptrdiff_t incLayOrigin,
			T* dest,
			std::ptrdiff_t incRowDest, std::ptrdiff_t incColDest, std::ptrdiff_t incLayDest
		  ) {
	// orig and origin denotes the original mesh data
	// highest value and corresponding order parameter
	// id are stored at places denoted by dest
	
	// loop over each lattice site
	#pragma omp parallel for
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<k; ++l) {
	
				T sumsq = 0;
				
				// loop all the values at each lattice site
				// loop size is determined by the value of
				// nonzero order parameters
				for (std::size_t u=0; u<noporig[i*incRowNoporig+j*incColNoporig+l*incLayNoporig]; ++u) {
					sumsq += origin[i*incRowOrigin+j*incColOrigin+l*incLayOrigin+u]
						   * origin[i*incRowOrigin+j*incColOrigin+l*incLayOrigin+u];
				}
				
				// store the value in buf
				dest[i*incRowDest+j*incColDest+l*incLayDest] = sumsq;
			}
		}
	}
}

/* wrapper function for C-style implementation
 * to call using mesh objects
*/
template <typename T, typename S>
void sumsq(mesh::thd::Mesh3d<S> &noporig, mesh::thd::Mesh3d<T> &origin, mesh::thd::Mesh3d<T> &dest) {
	// argument origin contains order parameter values
	// argument idmesh contains the corresponding ids
	// argument buf will have stored to id and the 
	// corresponding value
	
	// first check if dest and origin have the same
	// size
	assert(mesh::thd::check_compat(origin, dest));
	
	// call C-style implementation
	sumsq(noporig.xsize(), noporig.ysize(), noporig.zsize(),
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(), noporig.zinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(), origin.zinc(),
		   dest.ptr(),
		   dest.xinc(), dest.yinc(), dest.zinc());	
}
	

} /*namespace thd*/ } /*namespace mesh*/

#endif //SRC_MESH_SUMSQ3D_HPP
