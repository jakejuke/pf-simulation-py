#ifndef SRC_CUDA_DEVICEMESH2D_HPP
#define SRC_CUDA_DEVICEMESH2D_HPP 1

#ifdef __CUDACC__

#include <src/cuda/hostdev.hpp>
#include <src/cuda/components.hpp>
#include <cstddef>
#include <type_traits>
#include <cassert>

namespace cuda {

template <typename T>
class DeviceMesh2d :
    public Mesh2dDim,
    public DeviceData<T, false>,
    public Mesh2dElementAccess<DeviceMesh2d<T>>,
    public DeviceView<DeviceMesh2d<T>>
    {
        // delete default construtors,
        // assignement operators etc.
        DeviceMesh2d() = delete;
        DeviceMesh2d(const DeviceMesh2d&) = delete;
        DeviceMesh2d& operator=(const DeviceMesh2d) = delete;
        DeviceMesh2d& operator=(const DeviceMesh2d&&) = delete;

        public:
            
            // used for Require statements
            using is_DeviceMesh2d = std::true_type;
            using is_DeviceOnlyMesh2d = std::true_type;
        
            DeviceMesh2d(std::size_t m, std::size_t n, std::size_t maxnop,
                    StorageOrder2d order=StorageOrder2d::xy) :
                Mesh2dDim(m, n, maxnop, order),
                DeviceData<T, false> (m*n*maxnop)
            {
            }

            DeviceMesh2d(std::size_t m, std::size_t n,
                    StorageOrder2d order=StorageOrder2d::xy) :
                Mesh2dDim(m, n, 1, order),
                DeviceData<T, false> (m*n)
            {
            }

            DeviceMesh2d(DeviceMesh2d&&) = default;
            ~DeviceMesh2d() = default;
};


template <typename T>
class DeviceMesh2dView :
    public Mesh2dDim,
    public DeviceData<T, true>,
    public Mesh2dElementAccess<DeviceMesh2dView<T>>
    {

        public:
            // adjust default constructors and assignemnt
            // operators a view class accordingly
            DeviceMesh2dView() = delete;
            DeviceMesh2dView(const DeviceMesh2dView&) = default;
            DeviceMesh2dView& operator=(const DeviceMesh2dView) = delete;
            DeviceMesh2dView& operator=(const DeviceMesh2dView&&) = delete;

            // used for Require statements
            using is_DeviceMesh2d = std::true_type;
            using is_DeviceOnlyMesh2d = std::true_type;
            using is_DeviceView = std::true_type;
        
            HOST_DEV
            DeviceMesh2dView(std::size_t m, std::size_t n, std::size_t maxnop,
                    T* data,
                    ptrdiff_t incRow, ptrdiff_t incCol) :
                Mesh2dDim(m, n, maxnop, incRow, incCol),
                DeviceData<T, true> (data)
            {
            }

            DeviceMesh2dView(DeviceMesh2dView&&) = default;
            ~DeviceMesh2dView() = default;
};


// functions to easily create views
template <template <typename> class Mesh2dA, typename T>
HOST_DEV
DeviceMesh2dView<T> view_create(Mesh2dA<T> &A) {
    return DeviceMesh2dView<T>(A.xsize(), A.ysize(), A.opbufsize(), A.ptr(), A.xinc(), A.yinc());
}

} /* namespace cuda */


#else 
#   error This CUDA source must be compiled using nvcc
#endif


#endif

