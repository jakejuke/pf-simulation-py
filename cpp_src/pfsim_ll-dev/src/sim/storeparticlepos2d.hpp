#ifndef SRC_SIM_STOREPARTICLEPOS2D_HPP
#define SRC_SIM_STOERPARTICLEPOS2D_HPP

#include <src/aux/checkfileexists.hpp>
#include <cstddef>
#include <cstdio>

// this macro makes sure that we save the data
// to correct position; matlab uses index 
// starting from 1 but this program uses index zero
#ifndef START_INDEX
#define START_INDEX 1
#endif

namespace sim {
	
// function to store positions
// of particles that are used for
// "particle pinning" model
template <typename T>
int store_particle_pos2d(char filename[],
					std::size_t nof_particles, 
					T* posx, std::ptrdiff_t incPosx,
					T* posy, std::ptrdiff_t incPosy,
					bool overwrite=false
					) {
	
	// check if file alread exists
	if (!overwrite && aux::checkfileexists(filename)) {
		std::printf("Error when trying to write file\nFile %s exists\n",filename);
		return -1;
	}
	
	//open file
	FILE *fil = fopen(filename,"w+");
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	//init return value
	int lineswritten = 0;
	
	// loop over all particles
	for (std::size_t p=0; p<nof_particles; ++p) {
		std::fprintf(fil, "%ld	%ld\n", 
					posx[p*incPosx]+START_INDEX, 
					posy[p*incPosy]+START_INDEX);
		// increment line counter
		lineswritten++;
	}
	
	fclose(fil);
	
	return lineswritten;
}
	
} /* namespace sim */

# endif
