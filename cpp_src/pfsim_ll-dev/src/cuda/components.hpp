#ifndef SRC_CUDA_COMPONENTS_HPP
#define SRC_CUDA_COMPONENTS_HPP 1

#ifdef __CUDACC__

#include <src/cuda/devicebuffer.hpp>
#include <src/cuda/views.hpp>
#include <cstddef>
#include <cassert>

namespace cuda {

enum class StorageOrder2d{
    xy, /* row major */ 
    yx /* col major */
};


// class to manage dimensions
struct Mesh2dDim {

    const std::size_t numRows_, numCols_, maxnop_;
    const std::ptrdiff_t incRow_, incCol_;

    HOST_DEV
    Mesh2dDim(std::size_t m, std::size_t n, std::size_t maxnop,
            StorageOrder2d order) :
        numRows_(m), numCols_(n),
        maxnop_(maxnop),
        incRow_(order==StorageOrder2d::xy? numCols_*maxnop : maxnop),
        incCol_(order==StorageOrder2d::xy? maxnop : numRows_*maxnop)
    {
    }

    HOST_DEV
    Mesh2dDim(std::size_t m, std::size_t n, std::size_t maxnop,
            std::ptrdiff_t incRow, std::ptrdiff_t incCol) :
        numRows_(m), numCols_(n),
        maxnop_(maxnop),
        incRow_(incRow), incCol_(incCol)
    {
    }

    HOST_DEV
    std::size_t xsize() const {
        return numRows_;
    }
    
    HOST_DEV
    std::size_t ysize() const {
        return numCols_;
    }

    HOST_DEV
    std::size_t opbufsize() const {
        return maxnop_;
    }

    HOST_DEV
    std::ptrdiff_t xinc() const {
        return incRow_;
    }

    HOST_DEV
    std::ptrdiff_t yinc() const {
        return incCol_;
    }
};

// class to manage data
template <typename T, bool is_view>
struct DeviceData; 

template <typename T> 
struct DeviceData<T, true> {

    T* const data_;

    HOST_DEV
    DeviceData(T* data) : data_(data) {
    }

    HOST_DEV
    T* ptr() const {
        return data_;
    }
};

template <typename T> 
struct DeviceData<T, false> {    
    DeviceBuffer<T> buffer;

    DeviceData(std::size_t size) : buffer(size) {
    }

    HOST_DEV
    T* ptr() const {
#   ifdef __CUDA__ARCH__
        return nullptr;
#   else
        return buffer.ptr();
#   endif
    }
};

// class to allow elementwise access
template <typename Derived>
struct Mesh2dElementAccess {
#   ifdef __CUDACC__    
    
    // small hack to allow to call 
    // methods that are not part of this
    // very struct
    __device__
    const Derived& This() const {
        return *static_cast<const Derived*>(this);
    }

    __device__
    Derived& This() {
        return *static_cast<Derived*>(this);
    }

    // access of zeroth element
    __device__
    auto& operator()(std::size_t i, std::size_t j) {
        return This().ptr()[i*This().xinc()+j*This().yinc()];
    }

    __device__
    const auto& operator()(std::size_t i, std::size_t j) const {
        return This().ptr()[i*This().xinc()+j*This().yinc()];
    }

    // access of any other element
    __device__
    auto& operator()(std::size_t i, std::size_t j, std::size_t op) {
        return This().ptr()[i*This().xinc()+j*This().yinc()+op];
    }

    __device__
    const auto& operator()(std::size_t i, std::size_t j, std::size_t op) const {
        return This().ptr()[i*This().xinc()+j*This().yinc()+op];
    }
#   endif
};

template <typename Derived>
struct DeviceView {
    HOST_DEV
        Derived& This() {
            return *static_cast<Derived*>(this);
        }

    HOST_DEV
        const Derived& This() const {
            return *static_cast<const Derived*>(this);
        }

    template<typename... Args>
    HOST_DEV
    auto view(Args... args) {
        return cuda::view(This(), args...);
    }

    template<typename... Args>
    HOST_DEV
    auto view(Args... args) const {
        return cuda::view(This(), args...);
    }
};


} /* namespace cuda */


#else
#   error This CUDA source must be compiled using nvcc
#endif


#endif


