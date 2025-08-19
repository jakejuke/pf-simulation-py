#ifndef SRC_CUDA_MAXOP2D_HPP
#define SRC_CUDA_MAXOP2D_HPP 1

#ifdef __CUDACC__

#include <cstddef>

namespace cuda {

/* this stuff here is an implementation in C-Style, potentially fast
 * but generally prone to errors
template <typename ValType, typename IdType, typename NopType>    
__global__ void maxop2d(
			const NopType* __restrict__ noporig,
			std::ptrdiff_t incRowNoporig, std::ptrdiff_t incColNoporig,
			const ValType* __restrict__ origin,
			std::ptrdiff_t incRowOrigin, std::ptrdiff_t incColOrigin,
			const IdType* __restrict__ idorig,
			std::ptrdiff_t incRowIdorig, std::ptrdiff_t incColIdorig,
			IdType* iddest,
			std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest) {

    // get position in grid and ids
    std::size_t i = blockIdx.x;
    std::size_t j = blockIdx.y;
    std::size_t me = threadIdx.x;

    // setup shared memory used for reduction
    extern __shared__ ValType valmem[];

    // initialize shared memory
    if (me < noporig[i*incRowNoporig + j*incColNoporig]) {
        valmem[me] = origin[i*incRowOrigin+j*incColOrigin+me];
    } else {
        valmem[me] = 0;
    }

    // start off with reduction
    for (std::size_t active = blockDim.x / 2; active > 0; active /= 2) {
        __syncthreads(); 
        if (me < active) {
            // find max value in comparison and store 
            valmem[me] = max(valmem[me], valmem[me+active]);
        }
    }
    
    // now compare this maximal value to the array again
    // and store the id value where the value is equal to maximum
    if (valmem[0] == origin[i*incRowOrigin+j*incColOrigin+me]) {
        iddest[i*incRowIddest+j*incColIddest] = idorig[i*incRowIdorig+j*incColIdorig+me];
    }
}

// wrapper function
template <template<typename> class ValMesh,
         template<typename> class IdMesh,
         template<typename> class NopMesh,
         typename ValType,
         typename IdType,
         typename NopType>
void maxop2d(NopMesh<NopType> &noporig, ValMesh<ValType> &origin, IdMesh<IdType> &idorig,
        IdMesh<IdType> &iddest) {

    // define kernel configuration and size of shared mem
    dim3 blockconfig(idorig.opbufsize());
    dim3 gridconfig(noporig.xsize(), noporig.ysize());
    std::size_t memsize = origin.opbufsize()*sizeof(ValType);
    
    maxop2d<<<gridconfig, blockconfig, memsize>>>(
		   noporig.ptr(),
		   noporig.xinc(), noporig.yinc(),
		   origin.ptr(),
		   origin.xinc(), origin.yinc(),
		   idorig.ptr(),
		   idorig.xinc(), idorig.yinc(),
		   iddest.ptr(),
		   iddest.xinc(), iddest.yinc());
}
*/

/* implementation using aggregation techniques,
 * worse in performance, however
 
template <template<typename> class ValMesh,
         template<typename> class IdMesh,
         template<typename> class NopMesh,
         typename ValType,
         typename IdType,
         typename NopType>
__global__ void maxop2d(NopMesh<NopType> noporig, ValMesh<ValType> origin, IdMesh<IdType> idorig,
        IdMesh<IdType> iddest) {
    // get position in grid and ids
    std::size_t i = blockIdx.x;
    std::size_t j = blockIdx.y;
    std::size_t me = threadIdx.x;

    // setup shared memory used for reduction
    extern __shared__ ValType valmem[];

    // initialize shared memory
    if (me < noporig(i,j)) {
        valmem[me] = origin(i,j,me);
    } else {
        valmem[me] = 0;
    }

    // start off with reduction
    for (std::size_t active = blockDim.x / 2; active > 0; active /= 2) {
        __syncthreads(); 
        if (me < active) {
            // find max value in comparison and store 
            valmem[me] = max(valmem[me], valmem[me+active]);
        }
    }
    
    // now compare this maximal value to the array again
    // and store the id value where the value is equal to maximum
    if (valmem[0] == origin(i,j,me)) {
        iddest(i,j) = idorig(i,j,me);
    }
}
*/

// ultimate implementation
// grid and block configuration has to cover the complete mesh,
// e.g.:
// std::size_t blockdim = 4;
// dim3 gridconfigalt((m+blockdim-1)/blockdim, (n+blockdim-1)/blockdim);
// dim3 blockconfigalt(blockdim, blockdim);
template <template<typename> class ValMesh,
         template<typename> class IdMesh,
         template<typename> class NopMesh,
         typename ValType,
         typename IdType,
         typename NopType>
__global__ void maxop2d_alt(NopMesh<NopType> noporig, ValMesh<ValType> origin, IdMesh<IdType> idorig,
        IdMesh<IdType> iddest) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    ValType maxval = 0;
    IdType maxid = 0;

    if (i < noporig.xsize() && j < noporig.ysize()) {
        for (std::size_t nl=0; nl < noporig(i,j); ++nl) {
            if (origin(i,j,nl) > maxval) {
                maxval = origin(i,j,nl);
                maxid = idorig(i,j,nl);
            }
        }
        iddest(i,j) = maxid;
    }
}



} /* namespace cuda */


#else 
#   error This CUDA source must be compiled using nvcc
#endif


#endif

