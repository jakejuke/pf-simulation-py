#ifndef SRC_AUX_BUFFER_HPP
#define SRC_AUX_BUFFER_HPP

#include <cstddef>
#include <cassert>
#include <src/aux/alignmem.hpp>

// cache alignment macro
// should be equal to cacheline
// size or half of it
#ifndef CACHE_LINE_SIZE_DEFAULT
#define CACHE_LINE_SIZE_DEFAULT 64
#endif

namespace aux {
	
	
/* class for a standard buffer
 * to elegantly handle memory allocation
 * and freeing
 */
template <typename T>
class Buffer {
	
public:
	//allocate memory in constructor
	Buffer(std::size_t length) : length(length), mem(new T[aux::compute_aligned_size<T>(length, CACHE_LINE_SIZE_DEFAULT)]()), data(aux::align_ptr(mem, CACHE_LINE_SIZE_DEFAULT)) {
	}
	
	/* free memory upon destruction
	 *of the object
	 */	
	~Buffer() {
		delete[] mem;
	}
	
	// overload operator for convenient access
	T& operator[](std::size_t i) {
		return data[i];
	}
	
	/* simple method to set value
	 * inuniformly in the buffer*/
	void memset(T value) {
		for (std::size_t i=0; i<length; ++i) {
			mem[i] = value;
		}
	}
	
	// delete copy constructor and 
	// assignment operator to prevent
	// double free etc.
	Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;
	
	
	/* attributes needed to define 
	 * the buffer
	 */
	std::size_t length;
	T* mem;
	T* data;
};
	
	
} /*namespace aux*/

#endif
