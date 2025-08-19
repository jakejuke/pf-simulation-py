#ifndef SRC_AUX_LOADARRAY_HPP
#define SRC_AUX_LOADARRAY_HPP

#include <cstddef>
#include <cstdio>

namespace aux {
	
template <typename T>
int load_array(const char filename[], 
				std::size_t nof_elements,
				T* dest) {
	
	//open file
	std::FILE *fil = std::fopen(filename,"rb");
	
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file to load array\nCheck for path\n");
		return -1;
	}
	
	// read from file
	auto linesread = std::fread(dest, sizeof(T), nof_elements, fil);
	
	std::fclose(fil);
	
	return linesread;
}
	
} /*namespace aux*/

#endif
