#include <src/cuda/mesh2d.hpp>
#include <src/cuda/copy2d.hpp>
#include <src/cuda/hostdev.hpp>
#include <src/mesh/mesh2d.hpp>
#include <iostream>
#include <cstddef>

namespace aux {
HOST_DEV
inline std::size_t ringproject(int input, int ring_size);

}

#include <src/aux/ringproject.hpp>

template <template <typename> class Mesh, typename T>
__global__ void init(Mesh<T> m) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < m.xsize() && j < m.ysize()) {
        m(i,j,0) = i*10 + j;
    }
}

int main() {
    std::size_t m, n, k;
    m = n = k = 16;

    std::size_t blockdim = 4;
    std::size_t blocknum = (m+blockdim-1 ) / blockdim;

    std::cout << "some message" << std::endl;
    cuda::DeviceMesh2d<int> dmesh(m,n,k,cuda::StorageOrder2d::xy);

    mesh::twd::Mesh2d<int> hmesh(m,n,k,mesh::twd::StorageOrder2d::xy);
    cuda::copy(hmesh, dmesh);

    dim3 gridconfig(blocknum, blocknum);
    dim3 blockconfig(blockdim, blockdim);
    init<<<gridconfig, blockconfig>>>(dmesh.view());


    cuda::copy(dmesh, hmesh);

    for (std::size_t i=0; i<hmesh.xsize(); ++i) {
        for (std::size_t j=0; j<hmesh.ysize(); ++j) {
            std::cout << hmesh(i,j,0) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "end of script" << std::endl;
}
