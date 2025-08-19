#ifndef SRC_AUX_WALLTIME_HPP
#define SRC_AUX_WALLTIME_HPP 1

#include <chrono>

namespace aux {

template <typename T>
struct WallTime
{
    void
    tic()
    {
        t0 = std::chrono::high_resolution_clock::now();
    }

    T
    toc()
    {
        using namespace std::chrono;

        elapsed = high_resolution_clock::now() - t0;
        return duration<T,seconds::period>(elapsed).count();
    }

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::duration   elapsed;
};

} /*namespace aux*/


#endif // SRC_AUX_WALLTIME_HPP
