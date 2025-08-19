#ifndef SRC_AUX_SAVEARRAY_HPP
#define SRC_AUX_SAVEARRAY_HPP

#include <cstddef>
#include <cstdio>

namespace aux {
	
template <typename T>
int save_array(const char filename[], 
				std::size_t nof_elements,
				T* input) {
	
	//open file
	std::FILE *fil = fopen(filename,"wb");
	
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file to save array\nCheck for path\n");
		return -1;
	}
	
	// basically dump all the information into
	// the file
	auto lineswritten = fwrite(input, sizeof(T), nof_elements, fil);
	
	std::fclose(fil);
	
	return lineswritten;
}
	
} /*namespace aux*/

#endif
