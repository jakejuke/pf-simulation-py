#ifndef SRC_CUDA_COPY2D_HPP
#define SRC_CUDA_COPY2D_HPP 1

// TODO: update with c++20 standard
// that allows concepts

#ifdef __CUDACC__

#include <src/cuda/mesh2d.hpp>
#include <src/cuda/check.hpp>
#include <src/mesh/mesh2d.hpp>
#include <src/cuda/traits.hpp>
#include <cassert>

namespace cuda {

// function to check if current object
// has data stored consecutively
template <template<typename> class Mesh2dA, typename T>
bool consecutively_stored(const Mesh2dA<T>& m) {
    return m.xinc() == m.opbufsize() && m.yinc() == m.xsize()*m.opbufsize()
        || m.yinc() == m.opbufsize() && m.xinc() == m.ysize()*m.opbufsize();
}

// this is intended to work if such traits are added
// to the standard mesh2d function
/*
// copy host to device
template <
         template <typename> class MeshA,
         template <typename> class MeshB,
         typename T,
         traits::Require< 
            traits::DeviceMesh<MeshB<T>>,
            traits::HostOnlyMesh<MeshA<T>> > = true
> 
void copy(MeshA<T>& A, MeshB<T>& B) {
    assert(A.xsize() == B.xsize() && A.ysize() == B.ysize()
            && A.xinc() == B.xinc() && A.yinc() == B.yinc()
            && consecutively_stored(A) && consecutively_stored(B));

    CHECK_CUDA(cudaMemcpy, B.ptr(), A.ptr(),
            A.xsize()*A.ysize()*A.opbufsize()*sizeof(T),
            cudaMemcpyHostToDevice);
}

// copy device to host
template <
         template <typename> class MeshA,
         template <typename> class MeshB,
         typename T,
         traits::Require< 
            traits::DeviceMesh<MeshA<T>>,
            traits::HostOnlyMesh<MeshB<T>> > = true
> 
void copy(MeshA<T>& A, MeshB<T>& B) {
    assert(A.xsize() == B.xsize() && A.ysize() == B.ysize()
            && A.xinc() == B.xinc() && A.yinc() == B.yinc()
            && consecutively_stored(A) && consecutively_stored(B));

    CHECK_CUDA(cudaMemcpy, B.ptr(), A.ptr(),
            A.xsize()*A.ysize()*A.opbufsize()*sizeof(T),
            cudaMemcpyDeviceToHost);
}
*/

// explicit declaration
// device -> host
template <typename T>
void copy(cuda::DeviceMesh2d<T>& A, mesh::twd::Mesh2d<T>& B) {
    // check for compatibility
    assert(A.xsize() == B.xsize() && A.ysize() == B.ysize() 
            && A.xinc() == B.xinc() && A.yinc() == B.yinc() 
            && consecutively_stored(A) && consecutively_stored(B));

    // now copy stuff
    CHECK_CUDA(cudaMemcpy, B.ptr(), A.ptr(),
            A.xsize()*A.ysize()*A.opbufsize()*sizeof(T),
            cudaMemcpyDeviceToHost);
}

// host -> device
template <typename T>
void copy(mesh::twd::Mesh2d<T>& A, cuda::DeviceMesh2d<T>& B) {
    // check for compatibility
    assert(A.xsize() == B.xsize() && A.ysize() == B.ysize() 
            && A.xinc() == B.xinc() && A.yinc() == B.yinc() 
            && consecutively_stored(A) && consecutively_stored(B));

    // now copy stuff
    CHECK_CUDA(cudaMemcpy, B.ptr(), A.ptr(),
            A.xsize()*A.ysize()*A.opbufsize()*sizeof(T),
            cudaMemcpyHostToDevice);
}

} /* namespace cuda */


#else 
#   error This CUDA source must be compiled using nvcc
#endif


#endif

