#ifndef SRC_MESH_MAX2D_HPP
#define SRC_MESH_MAX2D_HPP 1

#include <cstddef>
#include <cassert>

namespace mesh {
	
// function to find the maximal number
// of nonzero op
template <template<typename> class Mesh, typename T>
T max2d(Mesh<T>& nop) {
	
	// initialize value that we will store max to
	T maxnop = 0;
	
	// check for storage organization
	if (nop.xinc()>nop.yinc()) {
		// run over each point and store the maximum,
		// omp manages the parallelization and reduction
		#pragma omp parallel for reduction(max:maxnop)
		for (std::size_t i=0; i<nop.xsize(); ++i) {
			for (std::size_t j=0; j<nop.ysize(); ++j) {
				if (nop(i,j) > maxnop) {
					maxnop = nop(i,j);
				}
			}
		}
	} else {
		#pragma omp parallel for reduction(max:maxnop)
		for (std::size_t j=0; j<nop.ysize(); ++j) {
			for (std::size_t i=0; i<nop.xsize(); ++i) {
				if (nop(i,j) > maxnop) {
					maxnop = nop(i,j);
				}
			}
		}
	}
	
	return maxnop;
}

} /*namespace mesh*/

#endif //SRC_MESH_MAXOP2D_HPP
