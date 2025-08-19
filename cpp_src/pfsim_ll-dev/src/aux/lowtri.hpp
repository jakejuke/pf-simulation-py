#ifndef SRC_AUX_LOWTRI_HPP
#define SRC_AUX_LOWTRI_HPP

#include <cstddef>
#include <cassert>

namespace aux {
	
	
/* class for a standard quadratic
 * lower triangular matrix
 * to elegantly handle memory allocation
 * and freeing; diagonal values are
 * assumed to be all 1 and, therefore,
 * are not stored here explicitly
 */
template <typename T>
class LowerTriangular {
	
public:
	//allocate memory in constructor,
	//square matrix is assumed here
	LowerTriangular(std::size_t dim) : dim(dim), length(dim*(dim-1)/2), data(new T[length]()) {
	}
	
	/* free memory upon destruction
	 *of the object
	 */	
	~LowerTriangular() {
		delete[] data;
	}
	
	/* simple method to set value
	 * inuniformly in the buffer*/
	void set(T value) {
		for (std::size_t i=0; i<length; ++i) {
			data[i] = value;
		}
	}
	
	/* access method:
	 * usage will be inefficient but
	 * it will suffice to look the 
	 * access pattern up;
	 * start off with read-only method
	 */
	const T &operator() (std::size_t i, std::size_t j) const {
		assert(i>=0 && i<dim);
		assert(j>=0 && j<dim);
		/* case i=j has to be handled 
		 * outside the function
		 */
		assert(i!=j);
		
		// case: j>i
		if (j>i) return data[j*(j-1)/2+i];
		// case:
		return data[i*(i-1)/2+j];
	}
	
	// write method
	T &operator() (std::size_t i, std::size_t j) {
		assert(i>=0 && i<dim);
		assert(j>=0 && j<dim);
		/* case i=j has to be handled 
		 * outside the function
		 */
		assert(i!=j);
		
		// case: j>i
		if (j>i) return data[j*(j-1)/2+i];
		// case:
		return data[i*(i-1)/2+j];
	}
	
	
	/* attributes needed to define 
	 * the buffer; also store dim
	 * although we usually don't
	 * need it
	 */
	std::size_t dim;
	std::size_t length;
	T* data;
};
	
	
} /*namespace aux*/

#endif
