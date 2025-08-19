#include <src/cuda/mesh2d.hpp>
#include <src/cuda/copy2d.hpp>
#include <src/cuda/hostdev.hpp>
#include <src/cuda/maxop2d.hpp>
#include <src/mesh/mesh2d.hpp>
#include <iostream>
#include <cstddef>

namespace aux {
HOST_DEV
inline std::size_t ringproject(int input, int ring_size);

}

#include <src/aux/ringproject.hpp>

template <template <typename> class Mesh, typename T, typename R, typename S>
__global__ void init(Mesh<T> nop, Mesh<R> val, Mesh<S> id) {
    std::size_t i = blockIdx.x;
    std::size_t j = blockIdx.y;
    std::size_t n = threadIdx.x;

    if (i < val.xsize() && j < val.ysize()) {
        val(i,j,n) = i*100 + j*10 + n;
        id(i,j,n) = n+1;
        nop(i,j) = blockDim.x;
    }
}

int main() {
    std::size_t m, n, k;
    m = n = 900;
    k = 32;

    auto dorder = cuda::StorageOrder2d::yx;
    auto horder = mesh::twd::StorageOrder2d::yx;

    std::cout << "some message" << std::endl;
    cuda::DeviceMesh2d<int> dval(m,n,k,dorder);
    cuda::DeviceMesh2d<int> dnop(m,n,dorder);
    cuda::DeviceMesh2d<int> did(m,n,k,dorder);
    cuda::DeviceMesh2d<int> diddest(m,n,dorder);

    mesh::twd::Mesh2d<int> hidmax(m,n,horder);

    // init on the device
    dim3 gridconfig(m, n);
    dim3 blockconfig(k);
    init<<<gridconfig, blockconfig>>>(dnop.view(), dval.view(), did.view());


    // now find maximum
    //cuda::maxop2d(dnop, dval, did, diddest);
    //cuda::maxop2d<<<gridconfig, blockconfig, sizeof(int)*did.opbufsize()>>>(dnop.view(), dval.view(), did.view(), diddest.view());
    std::size_t blockdim = 4;
    dim3 gridconfigalt((m+blockdim-1)/blockdim, (n+blockdim-1)/blockdim);
    dim3 blockconfigalt(blockdim, blockdim);
    cuda::maxop2d_alt<<<gridconfigalt, blockconfigalt>>>(dnop.view(), dval.view(), did.view(), diddest.view());

    // copy result
    cuda::copy(diddest, hidmax);

    for (std::size_t i=0; i<10; ++i) {
        for (std::size_t j=0; j<10; ++j) {
            std::cout << hidmax(i,j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "end of script" << std::endl;
}
