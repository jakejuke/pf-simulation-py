#ifndef SRC_SIM_GENORIENTATION_HPP
#define SRC_SIM_GENORIENTATION_HPP 1

#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <src/aux/buffer.hpp>

#ifndef INCORI
#define INCORI 9
#endif

#ifndef PI
#define PI 3.1415926
#endif

namespace sim {
	
template <typename T>
void gen_orientation(std::size_t num, T* dest, int seed=1) {
	
	// set seed
	std::srand(seed);
	
	// discard the first three values 
	// generated to match 
	// the original code
	T a = (T) (std::rand() + std::rand() + std::rand());
	// suppress compiler warning
	
	
	/* calculate the orientation matrix
		* for each position as given on:
		* https://de.qaz.wiki/wiki/Misorientation#Example_of_calculating_misorientation
		*/
	T b, c;
	for (std::size_t i=0; i<num; ++i) {
		a = ((T) std::rand() / RAND_MAX * 360) * PI / 180.;
		b = acos((T) (std::rand()*2-1)/ RAND_MAX);
		c = ((T) std::rand() / RAND_MAX * 360) * PI / 180.;
		
		dest[i*INCORI]  =cos(a)*cos(c)-sin(a)*sin(c)*cos(b);
		dest[i*INCORI+1]=sin(a)*cos(c)+cos(a)*sin(c)*cos(b);
		dest[i*INCORI+2]=sin(c)*sin(b);

		dest[i*INCORI+3]=-cos(a)*sin(c)-sin(a)*cos(c)*cos(b);
		dest[i*INCORI+4]=-sin(a)*sin(c)+cos(a)*cos(c)*cos(b);
		dest[i*INCORI+5]=cos(c)*sin(b);

		dest[i*INCORI+6]=sin(a)*sin(b);
		dest[i*INCORI+7]=-cos(a)*sin(b);
		dest[i*INCORI+8]=cos(b);
	}
}

// Wrapper function
template <typename T>
void gen_orientation(std::size_t num, 
					  aux::Buffer<T> &dest, 
					  int seed=1) {
	
	assert(num*INCORI==dest.length);
	
	gen_orientation(num, dest.data, seed);
}

} /*namespace sim*/


#endif
