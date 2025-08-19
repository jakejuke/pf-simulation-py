#include <cstddef>
#include <cstdio>

__global__ void axpy(double alpha, double* x, double* y) {
    std::size_t tid = threadIdx.x;
    y[tid] += alpha * x[tid];
}



int main () {

    // allocate memory on host
    std::size_t N = 16;
    double* x = new double[N];
    double* y = new double[N];

    // init vectors
    for (std::size_t i=0; i<N; ++i) {
        x[i] = i+1;
        y[i] = i+1;
    }

    // allocate memory on device
    double *cuda_x;
    cudaMalloc( (void**)&cuda_x, N*sizeof(double));
    double *cuda_y;
    cudaMalloc( (void**)&cuda_y, N*sizeof(double));

    // copy data to device
    cudaMemcpy( cuda_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( cuda_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

    // invoke kernel function 
    axpy<<<1, N>>> (2.0, cuda_x, cuda_y);

    // copy results back to host
    cudaMemcpy( x, cuda_x, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( y, cuda_y, N*sizeof(double), cudaMemcpyDeviceToHost);

    // release memory on device
    cudaFree( cuda_x);
    cudaFree( cuda_y);

    //print results
    std::printf("y = \n");
    for (std::size_t i=0; i<N; ++i) {
        std::printf("%lf\n", y[i]);
    }

    // release memory on host
    delete[] x;
    delete[] y;
}

