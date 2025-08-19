#ifndef SRC_MESH_MAXOP3D_HPP
#define SRC_MESH_MAXOP3D_HPP 1

#include <src/mesh/mesh3d.hpp>
#include <src/mesh/compat3d.hpp>
#include <cstddef>
#include <cassert>

namespace mesh { namespace thd {
	
/* function to extract value of maximal
 * order parameter at each lattice site
 * and its corresponding id
*/
template <typename T, typename S, typename R>
void maxop(std::size_t m, std::size_t n, std::size_t k,
			S* noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig, std::ptrdiff_t incLayNoporig,
			T* origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin, std::ptrdiff_t incLayOrigin,
			R* idorig,
			std::ptrdiff_t incRowIdorig, std::ptrdiff_t incColIdorig, std::ptrdiff_t incLayIdorig,
			T* dest,
			std::ptrdiff_t incRowDest, std::ptrdiff_t incColDest, std::ptrdiff_t incLayDest,
			R* iddest,
			std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest, std::ptrdiff_t incLayIddest
		  ) {
	// orig and origin denotes the original mesh data
	// highest value and corresponding order parameter
	// id are stored at places denoted by dest
	
	// loop over each lattice site
	#pragma omp parallel for
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<k; ++l) {
	
				T maxval = 0;
				T maxid = 0;
				
				// loop all the values at each lattice site
				// loop size is determined by the value of
				// nonzero order parameters
				T currentvalue;
				for (std::size_t u=0; u<noporig[i*incRowNoporig+j*incColNoporig+l*incLayNoporig]; ++u) {
					currentvalue = origin[i*incRowOrigin+j*incColOrigin+l*incLayOrigin+u];
					if (maxval<currentvalue) {
						maxval = currentvalue;
						maxid = idorig[i*incRowIdorig+j*incColIdorig+l*incLayIdorig+u];
					}
				}
				
				// store the value in buf
				iddest[i*incRowIddest+j*incColIddest+l*incLayIddest] = maxid;
				dest[i*incRowDest+j*incColDest+l*incLayDest] = maxval;
			}
		}
	}
}

/* wrapper function for C-style implementation
 * to call using mesh objects
*/
template <typename T, typename S, typename R>
void maxop(mesh::thd::Mesh3d<S> &noporig, mesh::thd::Mesh3d<T> &origin, mesh::thd::Mesh3d<R> &idorig, mesh::thd::Mesh3d<T> &dest, mesh::thd::Mesh3d<R> &iddest) {
	// argument origin contains order parameter values
	// argument idmesh contains the corresponding ids
	// argument buf will have stored to id and the 
	// corresponding value
	
	// first check if dest and origin have the same
	// size
	assert(mesh::thd::check_compat(origin, dest));
	assert(mesh::thd::check_compat(idorig, iddest));
	
	// call C-style implementation
	maxop(noporig.xsize(), noporig.ysize(), noporig.zsize(),
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(), noporig.zinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(), origin.zinc(),
		   idorig.ptr(),
		   idorig.xinc(), idorig.yinc(), idorig.zinc(),
		   dest.ptr(),
		   dest.xinc(), dest.yinc(), dest.zinc(),
		   iddest.ptr(),
		   iddest.xinc(), iddest.yinc(), iddest.zinc());
	
}

/* function to extract value id
 * of maximal op at each lattice site
*/
template <typename T, typename S, typename R>
void maxop(std::size_t m, std::size_t n, std::size_t k,
			S* noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig, std::ptrdiff_t incLayNoporig,
			T* origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin, std::ptrdiff_t incLayOrigin,
			R* idorig,
			std::ptrdiff_t incRowIdorig, std::ptrdiff_t incColIdorig, std::ptrdiff_t incLayIdorig,
			R* iddest,
			std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest, std::ptrdiff_t incLayIddest
		  ) {
	// orig and origin denotes the original mesh data
	// highest value and corresponding order parameter
	// id are stored at places denoted by dest
	
	// loop over each lattice site
	#pragma omp parallel for
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<k; ++l) {
	
				T maxval = 0;
				T maxid = 0;
				
				// loop all the values at each lattice site
				// loop size is determined by the value of
				// nonzero order parameters
				T currentvalue;
				for (std::size_t u=0; u<noporig[i*incRowNoporig+j*incColNoporig+l*incLayNoporig]; ++u) {
					currentvalue = origin[i*incRowOrigin+j*incColOrigin+l*incLayOrigin+u];
					if (maxval<currentvalue) {
						maxval = currentvalue;
						maxid = idorig[i*incRowIdorig+j*incColIdorig+l*incLayIdorig+u];
					}
				}
				
				// store the value in buf
				iddest[i*incRowIddest+j*incColIddest+l*incLayIddest] = maxid;
			}
		}
	}
}

/* wrapper function for C-style implementation
 * to call using mesh objects
*/
template <typename T, typename S, typename R>
void maxop(mesh::thd::Mesh3d<S> &noporig, mesh::thd::Mesh3d<T> &origin, mesh::thd::Mesh3d<R> &idorig, mesh::thd::Mesh3d<R> &iddest) {
	// argument origin contains order parameter values
	// argument idmesh contains the corresponding ids
	// argument buf will have stored to id and the 
	// corresponding value
	
	// first check if dest and origin have the same
	// size
	assert(mesh::thd::check_compat(idorig, iddest));
	
	// call C-style implementation
	maxop(noporig.xsize(), noporig.ysize(), noporig.zsize(),
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(), noporig.zinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(), origin.zinc(),
		   idorig.ptr(),
		   idorig.xinc(), idorig.yinc(), idorig.zinc(),
		   iddest.ptr(),
		   iddest.xinc(), iddest.yinc(), iddest.zinc());
	
}
	

} /*namespace thd*/ } /*namespace mesh*/

#endif //SRC_MESH_MAXOP3D_HPP
