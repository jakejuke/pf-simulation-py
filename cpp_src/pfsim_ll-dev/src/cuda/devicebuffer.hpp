#ifndef SRC_CUDA_DEVICEBUFFER_HPP
#define SRC_CUDA_DEVICEBUFFER_HPP

#ifdef __CUDACC__

#include <src/cuda/check.hpp>
#include <src/cuda/hostdev.hpp>
#include <cstddef>
//#include <cstdint>

#ifndef DEVICE_CACHE_LINE_SIZE_DEFAULT
#define DEVICE_CACHE_LINE_SIZE_DEFAULT 64
#endif

namespace cuda {


// function that helps to compute size of
// allocated memory that includes an extra 
// cache line to account for chache line
// alignment
template<typename T>
std::size_t compute_aligned_size(std::size_t length, std::size_t wanted_alignment) {
    // TODO: prone to errors if wanted_alignment < alignof(T)
    // currently not supported by fractal
    return length * sizeof(T) + wanted_alignment;
}

// function that helps to determine
// aligned pointer
template<typename T>
T* align_ptr(const void* ptr, std::size_t wanted_alignment) {
    // TODO: fix this whole function to new standards
    // --> std::max, std::uintptr
    std::size_t alignment = wanted_alignment; 
    //std::max(wanted_alignment, alignof(T));
    return (T*) (((std::size_t) ptr + alignment) & ~(alignment - 1));
}

// function that helps with memory allocation
// on the device
inline void* cuda_malloc(std::size_t size) {
    void* devptr;
    CHECK_CUDA(cudaMalloc, &devptr, size);
    return devptr;
}

// simple RAII struct to allocate
// memory upon construction and dealloctate
// upon destruction
template <typename T>
struct DeviceBuffer {

    DeviceBuffer(std::size_t length, std::size_t alignment=DEVICE_CACHE_LINE_SIZE_DEFAULT) :
        length(length),
        devmem(cuda_malloc(compute_aligned_size<T>(length, alignment))),
        devdata(align_ptr<T>(devmem, alignment)) {
    }

    ~DeviceBuffer() {
        CHECK_CUDA(cudaFree, devmem);
    }

    // get method for pointer
    HOST_DEV
    T* ptr() const {
        return devdata;
    }

    // delete copy constructor and
    // assignment operator and add
    // default "whatever this is good for"
    DeviceBuffer(DeviceBuffer&&) = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    std::size_t length;
    // pointer to memory and
    // aligned pointer, starting from
    // which the memory will be used
    void* const devmem;
    T* const devdata;
};


} /* namespace cuda */

#else
#   error This source must be compiled using nvcc
#endif

#endif // SRC_CUDA_DEVICEBUFFER_HPP
