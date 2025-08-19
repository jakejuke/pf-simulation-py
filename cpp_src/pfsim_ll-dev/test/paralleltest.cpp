#include <cstddef>
#include <cstdio>
#include <omp.h>
#include <src/aux/walltime.hpp>


int main() {

    const std::size_t size = 1e8;
    double* buf = new double[size];


    aux::WallTime<float> wt;
    wt.tic();

    for (std::size_t i=0; i<size; ++i) {
        buf[i] = i+1;
    }
    std::printf("time elapsed for single access: %f\n", wt.toc());

    wt.tic();
    #pragma omp parallel for
    for (std::size_t i=0; i<size; ++i) {
        buf[i] += 2;
    }
    std::printf("time elapsed for mp access: %f\n", wt.toc());

    wt.tic();
    for (std::size_t i=0; i<size; ++i) {
        buf[i] = i+1;
    }
    std::printf("time elapsed for second single access: %f\n", wt.toc());


    delete[] buf;
}
