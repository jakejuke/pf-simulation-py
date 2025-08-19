// 11.09.22 Julian Jakob
#ifndef SRC_SIM_SAVECOEFFMAP_HPP
#define SRC_SIM_SAVECOEFFMAP_HPP 1

#include <cstddef>
#include <cstdio>
#include <src/aux/buffer.hpp>


namespace sim {
	
template <typename T>
int save_coeffmap(const char filename[], std::size_t num, T* orig) {
	
	//open file
	std::FILE *fil = fopen(filename,"w+");
	
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file to save orientations\nCheck for path\n");
		return -1;
	}
	
	//init return value
	int lineswritten = 0;
	
	// run over all entries in batches of 9 values
	for (std::size_t i=0; i<num; ++i) {
		
		// write lines individually
		std::fprintf(fil, "%f\n",
					orig[i*2]);

		// increment counter
		++lineswritten;
	}
	
	std::fclose(fil);
	
	return lineswritten;
}

// overload to also accept buffers
// explicitly expects double buffer!
/*
int save_orientation(const char filename[], aux::Buffer<double> input) {
	return save_orientation(filename, input.length, input.data);
}
*/
	
} /*namespace sim*/ 


#endif