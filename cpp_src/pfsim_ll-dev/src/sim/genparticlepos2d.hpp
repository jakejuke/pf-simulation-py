#ifndef SRC_SIM_GENPARTICLEPOS2D_HPP
#define SRC_SIM_GENPARTICLEPOS2D_HPP

#include <cstddef>
#include <cstdlib>

namespace sim {
	
// function to generate positions of
// nof_particles in two dimensions,
// i.e. each particle needs x coordinate
// and each needs y coordinate;
// easiest call: take array of length 2*nof_particles,
// then pass ptr to zero element as posx and incPosx=2,
// for posy pass ptr to index 1 element and incPosy=2
template <typename T>
void gen_particle_pos2d(std::size_t m, std::size_t n,
					std::size_t nof_particles, 
					T* posx, std::ptrdiff_t incPosx,
					T* posy, std::ptrdiff_t incPosy,
					int seed=1) {
	
	// set seed
	std::srand(seed);
	
	// loop to create positions
	for (std::size_t i=0; i<nof_particles; ++i) {
		posx[i*incPosx] = static_cast<T>(std::rand()%m);
		posy[i*incPosy] = static_cast<T>(std::rand()%n);
        
        // from here on special case
        int center = 119;
        int offset_center = 70;
        if (i == 0){
            posx[i*incPosx] = static_cast<T>(center + offset_center);
            posy[i*incPosy] = static_cast<T>(center);
        }
        if (i == 1){
            posx[i*incPosx] = static_cast<T>(center);
            posy[i*incPosy] = static_cast<T>(center + offset_center);
        }
        if (i == 2){
            posx[i*incPosx] = static_cast<T>(center - offset_center);
            posy[i*incPosy] = static_cast<T>(center);
        }
        if (i == 3){
            posx[i*incPosx] = static_cast<T>(center);
            posy[i*incPosy] = static_cast<T>(center - offset_center);
        }
	}
}
	
} /* namespace sim */

# endif
