#ifndef SRC_MESH_MAXOP2D_HPP
#define SRC_MESH_MAXOP2D_HPP 1

#include <src/mesh/mesh2d.hpp>
#include <src/mesh/compat2d.hpp>
#include <cstddef>
#include <cassert>

namespace mesh { namespace twd {
	
/* function to extract value of maximal
 * order parameter at each lattice site
 * and its corresponding id
*/
template <typename T, typename S, typename R>
void maxop(std::size_t m, std::size_t n,
			S* noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig,
			T* origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin,
			R* idorig,
			std::ptrdiff_t incRowIdorig, std::ptrdiff_t incColIdorig,
			T* dest,
			std::ptrdiff_t incRowDest, std::ptrdiff_t incColDest,
			R* iddest,
			std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest) {
	// orig and origin denotes the original mesh data
	// highest value and corresponding order parameter
	// id are stored at places denoted by dest
	
	// loop over each lattice site
	#pragma omp parallel for
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
	
			T maxval = 0;
			T maxid = 0;
			
			// loop all the values at each lattice site
			// loop size is determined by the value of
			// nonzero order parameters
			T currentvalue;
			for (std::size_t u=0; u<noporig[i*incRowNoporig+j*incColNoporig]; ++u) {
				currentvalue = origin[i*incRowOrigin+j*incColOrigin+u];
				if (maxval<currentvalue) {
					maxval = currentvalue;
					maxid = idorig[i*incRowIdorig+j*incColIdorig+u];
				}
			}
			
			// store the value in buf
			iddest[i*incRowIddest+j*incColIddest] = maxid;
			dest[i*incRowDest+j*incColDest] = maxval;
		}
	}
}

/* wrapper function for C-style implementation
 * to call using mesh objects
*/
template <typename T, typename S>
void maxop(mesh::twd::Mesh2d<S> &noporig, mesh::twd::Mesh2d<T> &origin, mesh::twd::Mesh2d<S> &idorig, mesh::twd::Mesh2d<T> &dest, mesh::twd::Mesh2d<S> &iddest) {
	// argument origin contains order parameter values
	// argument idmesh contains the corresponding ids
	// argument buf will have stored to id and the 
	// corresponding value
	
	// first check if dest and origin have the same
	// size
	assert(mesh::twd::check_compat(origin, dest));
	assert(mesh::twd::check_compat(idorig, iddest));
	
	// call C-style implementation
	maxop(noporig.xsize(), noporig.ysize(),
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(),
		   idorig.ptr(),
		   idorig.xinc(), idorig.yinc(),
		   dest.ptr(),
		   dest.xinc(), dest.yinc(),
		   iddest.ptr(),
		   iddest.xinc(), iddest.yinc());
	
}

/* function to extract id of maximal
 * op at each lattice site
*/
template <typename T, typename S, typename R>
void maxop(std::size_t m, std::size_t n,
			S* noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig,
			T* origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin,
			R* idorig,
			std::ptrdiff_t incRowIdorig, std::ptrdiff_t incColIdorig,
			R* iddest,
			std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest) {
	// orig and origin denotes the original mesh data
	// highest value and corresponding order parameter
	// id are stored at places denoted by dest
	
	// loop over each lattice site
	#pragma omp parallel for
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
	
			T maxval = 0;
			T maxid = 0;
			
			// loop all the values at each lattice site
			// loop size is determined by the value of
			// nonzero order parameters
			T currentvalue;
			for (std::size_t u=0; u<noporig[i*incRowNoporig+j*incColNoporig]; ++u) {
				currentvalue = origin[i*incRowOrigin+j*incColOrigin+u];
				if (maxval<currentvalue) {
					maxval = currentvalue;
					maxid = idorig[i*incRowIdorig+j*incColIdorig+u];
				}
			}
			
			// store the value in buf
			iddest[i*incRowIddest+j*incColIddest] = maxid;
		}
	}
}

/* wrapper function for C-style implementation
 * to call using mesh objects
*/
template <typename T, typename S>
void maxop(mesh::twd::Mesh2d<S> &noporig, mesh::twd::Mesh2d<T> &origin, mesh::twd::Mesh2d<S> &idorig, mesh::twd::Mesh2d<S> &iddest) {
	// argument origin contains order parameter values
	// argument idmesh contains the corresponding ids
	// argument buf will have stored to id and the 
	// corresponding value
	
	// first check if dest and origin have the same
	// size
	assert(mesh::twd::check_compat(idorig, iddest));
	
	// call C-style implementation
	maxop(noporig.xsize(), noporig.ysize(),
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(),
		   idorig.ptr(),
		   idorig.xinc(), idorig.yinc(),
		   iddest.ptr(),
		   iddest.xinc(), iddest.yinc());
	
}
	

} /*namespace twd*/ } /*namespace mesh*/

#endif //SRC_MESH_MAXOP2D_HPP
