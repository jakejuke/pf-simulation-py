#ifndef SRC_SIM_READCOEFFMAPS_HPP 
#define SRC_SIM_READCOEFFMAPS_HPP 1

#include <cstdio>
#include <cassert>
#include <cstddef>

namespace sim {
	
//C-style implementation
template <typename T>
int readcoeffmaps(const char filename[], 
					T* dest, std::ptrdiff_t incDest) {
	
	FILE *fil = fopen(filename,"r");
	std::size_t linesread = 0;
	
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	
	// iterate over lines and process input
	// TODO: optimization by loading data in chunks
	// and processing it afterwards
	while(!feof(fil)) {
		
		//read line-by-line and check for errors
		if(std::fscanf(fil, "%f\n", &dest[linesread*incDest]) != 1) {
			std::printf("error when reading lines\nCheck for file content\n");
			return -1;
		}

		++linesread;
	}
	
	fclose(fil);
	
	return linesread;
}
					

	
} /*namespace sim*/ 

#endif // SRC_SIM_READCOEFFMAPS_HPP
