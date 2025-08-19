// 11.02.22 Julian Jakob
#ifndef SRC_AUX_LOADORI_HPP
#define SRC_AUX_LOADORI_HPP

#include <cstddef>
#include <cstdio>

#ifndef INCORI
#define INCORI 9
#endif



namespace aux {
	
template <typename T>
int load_ori(const char filename[], 
				std::size_t num,
				T* dest) {
	
	//open file
	std::FILE *fil = std::fopen(filename,"r");
	
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file to load ori\nCheck for path\n");
		return -1;
	}
	
	int linesread = 0;

	// run over all entries in batches of 9 values
	for (std::size_t i=0; i<num; ++i) {

	// read from file
	 		std::fscanf (fil, "%f %f %f %f %f %f %f %f %f\n", 
					&dest[i*INCORI], &dest[i*INCORI+1], &dest[i*INCORI+2],
					&dest[i*INCORI+3], &dest[i*INCORI+4], &dest[i*INCORI+5],
					&dest[i*INCORI+6], &dest[i*INCORI+7], &dest[i*INCORI+8]);
	
	linesread++;
	}

	std::fclose(fil);
	
	return linesread;
}
	
} /*namespace aux*/

#endif