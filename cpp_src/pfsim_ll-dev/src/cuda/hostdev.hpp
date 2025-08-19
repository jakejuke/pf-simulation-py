#ifndef HPC_CUDA_HOSTDEV_HPP
#define HPC_CUDA_HOSTDEV_HPP 1

/* HOST_DEV is used to mark functions or method declarations that
   can be compiled for the host and the device as well;
   this macro has no expansion if we do not compile using nvcc */

#ifdef __CUDACC__
#	define HOST_DEV __host__ __device__
#else
#	define HOST_DEV
#endif

#endif
