#ifndef HPC_CUDA_CHECK_HPP
#define HPC_CUDA_CHECK_HPP 1

#ifdef __CUDACC__

/* this header file defines a macro that can be used
   to wrap all CUDA library calls. Example:

      int device; cudaGetDevice(&device);

   can be replaced by

      CHECK_CUDA(cudaGetDevice, &device);

   Whenever an error is detected, CHECK_CUDA
   throws a hpc::cuda::exception object which
   delivers a helpful message pointing to the
   failed CUDA library call.
*/

#include <sstream>

namespace hpc { namespace cuda {

class exception: public std::exception {
   public:
      exception(const char* cudaop, const char* source,
	    int line, cudaError_t error) :
	 cudaop(cudaop), source(source), line(line), error(error) {
      }
      const char* what() const noexcept override {
	 message.clear();
	 std::ostringstream out;
	 out << cudaop << " at " << source << ":" << line << " failed: " << cudaGetErrorString(error); 
	 message = out.str();
	 return message.c_str();
      }
   private:
      const char* cudaop;
      const char* source;
      int line;
      cudaError_t error;
      mutable std::string message;
};

inline void check_error(const char* cudaop, const char* source,
       int line, cudaError_t error) {
   if (error != cudaSuccess) {
      throw exception(cudaop, source, line, error);
   }
}

} } // namespaces cuda, hpc

#define CHECK_CUDA(opname, ...) \
   hpc::cuda::check_error(#opname, __FILE__, __LINE__, opname(__VA_ARGS__))

#define CHECK_KERNEL \
   hpc::cuda::check_error("kernel", __FILE__, __LINE__, cudaGetLastError())

#else
#	error This CUDA source must be compiled using nvcc
#endif

#endif
