#ifndef SRC_CUDA_VIEWS_HPP
#define SRC_CUDA_VIEWS_HPP 1

#include <src/cuda/hostdev.hpp>

namespace cuda {

template <typename T>
HOST_DEV
auto view (T&& x) {
    return view_create(x);
}



} /* namespace cuda */

#endif
