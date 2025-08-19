#ifndef SRC_SIM_CALCMISORIENTATION_HPP
#define SRC_SIM_CALCMISORIENTATION_HPP

#include <cstddef>
#include <cassert>
#include <src/aux/lowtri.hpp>
#include <src/aux/buffer.hpp>

#ifndef INCORI
#define INCORI 9
#endif

#ifndef GRAIN_ID_START
#define GRAIN_ID_START 1
#endif

#ifndef PI
#define PI 3.1415926
#endif

namespace sim {

float symmetry[216]={
	 1,  0,  0,  0,  1,  0,  0,  0,  1,
	 0, -1,  0,  1,  0,  0,  0,  0,  1,
	-1,  0,  0,  0, -1,  0,  0,  0,  1,
	 0,  1,  0, -1,  0,  0,  0,  0,  1,

	 1,  0,  0,  0,  0, -1,  0,  1,  0,
	 1,  0,  0,  0, -1,  0,  0,  0, -1,
	 1,  0,  0,  0,  0,  1,  0, -1,  0,
	 0,  0,  1,  0,  1,  0, -1,  0,  0,

	-1,  0,  0,  0,  1,  0,  0,  0, -1,
	 0,  0, -1,  0,  1,  0,  1,  0,  0,
	 0,  0,  1,  1,  0,  0,  0,  1,  0,
	 0,  1,  0,  0,  0,  1,  1,  0,  0,

	 0, -1,  0,  0,  0, -1,  1,  0,  0,
	 0,  0,  1, -1,  0,  0,  0, -1,  0,
	 0, -1,  0,  0,  0,  1, -1,  0,  0,
	 0,  0, -1, -1,  0,  0,  0,  1,  0,

	 0,  0, -1,  1,  0,  0,  0, -1,  0,
	 0,  1,  0,  0,  0, -1, -1,  0,  0,
	 0,  1,  0,  1,  0,  0,  0,  0, -1,
	 0, -1,  0, -1,  0,  0,  0,  0, -1,

	-1,  0,  0,  0,  0,  1,  0,  1,  0,
	-1,  0,  0,  0,  0, -1,  0, -1,  0,
	 0,  0,  1,  0, -1,  0,  1,  0,  0,
	 0,  0, -1,  0, -1,  0, -1,  0,  0
	};

/* function to calculate the 
 * misorientation between grains 
 * with ids ida and idb, that start
 * from value 1; 
 * for array access pattern however,
 * we start from value 0, so that for
 * access we have to decrement ida
 * and idb by one;
 * TODO: incorporate mechanism to
 * prevent misusage, i.e. by too small
 * buffer
 */
template <typename T, typename R>
T calc_misorientation(R ida, R idb, 
						   T *orientation,
						   T *rotation_buffer) {

	/* multiply the orientation matrix of 
	 * grain a transposed with the matrix of
	 * grain b;
	 * here we have used the increments to 
	 * access orientation by having incRow=3
	 * and incCol=1; to access the transposed
	 * matrix simply swap these increments
	 * (see for access using ida;
	 * TODO: come up with sth. smart to make
	 * use of BLAS functions
	 */
	
	for (std::size_t i=0; i<3; ++i) {
		for (std::size_t j=0; j<3; ++j) {
			rotation_buffer[3*i+j] = 0;
			for (std::size_t l=0; l<3; ++l) {
					rotation_buffer[3*i+j] += 
					orientation[INCORI*(ida-GRAIN_ID_START)+i+3*l] 
					* orientation[INCORI*(idb-GRAIN_ID_START)+3*l+j];
			}
		}
	}

	
	
	/* now try all possible symmetry operations
	 * and determine trace after each;
	 * maximal trace is needed for return value;
	 * start off with main loop over all symmetries
	 */
	T maxtrace = 0;
	for (std::size_t n=0; n<24; ++n) {
		
		/* now two loops to obtain the trace
		 * of the matrix product; see:
		 * https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product
		 * for reference;
		 * TODO: optimization by usage of 
		 * dot product if data is arranged
		 * properly in memory
		 */
		T currenttrace = 0;
		for (std::size_t i=0; i<3; ++i) {
			for (std::size_t j=0; j<3; ++j) {
				currenttrace += rotation_buffer[3*i+j] * symmetry[INCORI*n+i+3*j];
			}
		}
		
		// check if trace is maximal
		if (currenttrace > maxtrace) maxtrace = currenttrace;
	}
	
	// return the desired value
	return acos((maxtrace-1) / 2) / PI * 180.;
}

/* function to calculate misorientation
 * between all available grain ids;
 * TODO: incorporate mechanis to prevent
 * misusage;
 * passing a lower triangular is an exception
 */	
template <typename T>
void calc_all_misorientation(std::size_t num, 
							 T* orientation,
							 T* misorientation) {
	
	
	/* run over all grain ids
	 * combinations (indexing from zero(
	 * and determine misorientation to 
	 * store it in the lower triangular 
	 * matrix object;
	 * outer loop starts with 1 because we
	 * do not want to calculate the values on
	 * the diagonal (i.e. between grains with the
	 * same id), j stops below i so that we never 
	 * determine the misorientation for i==j
	 */
	#pragma omp parallel for
	for (std::size_t i=1; i<num; ++i) {
		// allocate memory locall 
		// to pass as a buffer
		T rotation_buffer[9];
		for (std::size_t j=0; j<i; ++j) {
			misorientation[i*(i-1)/2+j] = 
			calc_misorientation(i+GRAIN_ID_START, 
								j+GRAIN_ID_START, 
								orientation,
								rotation_buffer);
		}
	}
}

//Wrapper function
template <typename T>
void calc_all_misorientation(std::size_t num, 
							 aux::Buffer<T> &orientation,
							 aux::LowerTriangular<T> &misorientation) {
	assert(num<=misorientation.dim);
	assert(orientation.length==misorientation.dim*INCORI);
	
	calc_all_misorientation(num,
							orientation.data,
							misorientation.data);

}


// function to get misorientation
/*
template <typename T, typename R>
T get_misorientation(R ida, R idb, T* orig) {

	// access orig to find the misorientation
	// between grains with ids ida and idb
}
*/
 
} /*namespace sim*/

#endif // SRC_SIM_CALCMISORIENTATION_HPP
