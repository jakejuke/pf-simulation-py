#ifndef SRC_CUDA_STEPISO2D_HPP
#define SRC_CUDA_STEPISO2D_HPP 1

#ifdef __CUDACC__

#include <cstddef>

#ifndef LCOEFF
#define LCOEFF 0.65595
#endif

#ifndef KAPPACOEFF
#define KAPPACOEFF 4.2686
#endif

#ifndef GAMMACOEFF
#define GAMMACOEFF 1.5
#endif

#ifndef MCOEFF
#define MCOEFF 8.5372
#endif

#ifndef DELTAT
#define DELTAT 0.015
#endif

#ifndef DELTAX
#define DELTAX 0.5
#endif

#ifndef THRESHOLD
#define THRESHOLD 1e-5
#endif

namespace cuda {

template <typename ValType, typename IdType, typename NopType>    
__global__ void stepiso2d(
            std::size_t izero, std::size_t jzero,
            std::size_t mparent, std::size_t nparent,
			const NopType* __restrict__ orignop, std::ptrdiff_t incRowOrignop, std::ptrdiff_t incColOrignop,
			const ValType* __restrict__ origval, std::ptrdiff_t incRowOrigval, std::ptrdiff_t incColOrigval,
			const IdType* __restrict__ origid, std::ptrdiff_t incRowOrigid, std::ptrdiff_t incColOrigid,
			NopType* destnop, std::ptrdiff_t incRowDestnop, std::ptrdiff_t incColDestnop,
			ValType* destval, std::ptrdiff_t incRowDestval, std::ptrdiff_t incColDestval,
			IdType* destid, std::ptrdiff_t incRowDestid, std::ptrdiff_t incColDestid
			) {

    // get position in grid and ids
    std::size_t i = blockIdx.x + izero;
    std::size_t j = blockIdx.y + jzero;
    std::size_t me = threadIdx.x;

    // setup shared memory used for reduction
    extern __shared__ IdType idmem[];
    
    // copy ids at current positions
    if (me < orignop[i*incRowOrignop+j*incColOrignop]) {
        idmem[me] = Origid[i*incRowOrigid+j*incColOrigId+me];
    } else {
        idmem[me] = 0;
    }

    // determine surrounding positions
    // while taking periodic boundary
    // conditions into account
    int positions[] = {
        i, (j+1) % nparent,
        i, (j-1+nparent) % nparent,
        (i+1) % mparent, j,
        (i-1+mparent) % mparent, j};

    // flag for finding value
    __shared__ fflag = 0;

    // value of totalnop
    __shared__ std::size_t totalnop = orignop[i*incRowOrignop+j*incColOrignop];

    // run over all nearest neighbors
    for (std::size_t nl=0; nl<8; nl+=2) {

        // for each neighbor run over each id
        for (std::size_t currentindex=0; 
                currentindex < orignop[pos[nl]*incRowOrignop+pos[nl+1]*incColOrignop];
                ++currentindex) {

            // check if th current id is present in idmem and add value if it is
            if (idmem[me] == origid[pos[nl]*incRowOrigid+pos[nl+1]*incColOrigId+currentindex] && me < totalnop) {
                destval[i*incRowDestval+j*incColDestval+me] += origval[pos[nl]*incRowOrigval+pos[nl+1]*incColOrigval+currentindex]
               ++fflag; 
            }

            // if id has not been found, add it to idmem
            if (fflag == 0 && currentindex == me) {
                idmem[totalnop] = origid[pos[nl]*incRowOrigid+pos[nl+1]*incColOrigId+currentindex];
                destval[totalnop] = origval[pos[nl]*incRowOrigval+pos[nl+1]*incColOrigval+currentindex];
                totalnop++;
            } 

            // reset flag 
            if (me==currentindex) {
                fflag = 0;
            }
        }
    }

    // check for overflow
    if (me == 0 && totalnop>blockDim.x) {
        return;
    }
    
    // proceed with the actual calculation for each order parameter
    if (me < totalnop) {
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



} /* namespace cuda */


#else 
#   error This CUDA source must be compiled using nvcc
#endif


#endif

