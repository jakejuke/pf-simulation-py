#ifndef SRC_AUX_ALIGNMEM_HPP
#define SRC_AUX_ALIGNMEM_HPP

#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace aux {
	
	
// function to compute the length of 
// of memory that needs to be allocated
// to have enough space to start inserting
// data where it is aligned to a specified
// alignment
template <typename T>
std::size_t compute_aligned_size(std::size_t length, std::size_t desired_alignment) {
	// basically add enough space 
	// to allow another complete cacheline
	// to fit in
	return length + desired_alignment / sizeof(T);
}


// function to determine the pointer inside
// already allocated memory, that is aligned
// to a specific alignment value
template <typename T>
T* align_ptr(T* memptr, std::size_t desired_alignment) {
	// for when fractal allows for newer standards
	//std::size_t alignment = std::max(alignof(T), desired_alignment);
	std::size_t alignment = desired_alignment;
	return (T*) (((std::uintptr_t) memptr + alignment) & ~(alignment - 1 ));
}


	
} /*namespace aux*/


#endif // SRC_AUX_ALIGNMEM_HPP
