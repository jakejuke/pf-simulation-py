#ifndef SRC_SIM_GENPARTICLEPOS3D_HPP
#define SRC_SIM_GENPARTICLEPOS3D_HPP 1

#include <cstddef>
#include <cstdlib>

namespace sim {
	
// function to generate positions of
// nof_particles in two dimensions,
// i.e. each particle needs x coordinate
// and each needs y and z  coordinate;
// easiest call: take array of length 2*nof_particles,
// then pass ptr to zero element as posx and incPosx=3,
// for posy pass ptr to index 1 element and incPosy=3,
// for posz pass ptr to index 2 element and incPosz=3
template <typename T>
void gen_particle_pos3d(std::size_t m, std::size_t n, std::size_t k,
					std::size_t nof_particles, 
					T* posx, std::ptrdiff_t incPosx,
					T* posy, std::ptrdiff_t incPosy,
					T* posz, std::ptrdiff_t incPosz,
					int seed=1) {
	
	// set seed
	std::srand(seed);
	
	// loop to create positions
	for (std::size_t i=0; i<nof_particles; ++i) {
//        not used for specific particle positions
		posx[i*incPosx] = static_cast<T>(std::rand()%m);
		posy[i*incPosy] = static_cast<T>(std::rand()%n);
		posz[i*incPosz] = static_cast<T>(std::rand()%n);
        
////        from here on special case
//        int center = 59;
////        R_Sphere = 50 -> offset_center < 48
//        int offset_center = 30;
//        if (i == 0){
//            posx[i*incPosx] = static_cast<T>(center + offset_center);
//            posy[i*incPosy] = static_cast<T>(center);
//            posz[i*incPosz] = static_cast<T>(center);
//        }
//        if (i == 1){
//            posx[i*incPosx] = static_cast<T>(center);
//            posy[i*incPosy] = static_cast<T>(center + offset_center);
//            posz[i*incPosz] = static_cast<T>(center);
//        }
//        if (i == 2){
//            posx[i*incPosx] = static_cast<T>(center);
//            posy[i*incPosy] = static_cast<T>(center);
//            posz[i*incPosz] = static_cast<T>(center + offset_center);
//        }
//        if (i == 3){
//            posx[i*incPosx] = static_cast<T>(center - offset_center);
//            posy[i*incPosy] = static_cast<T>(center);
//            posz[i*incPosz] = static_cast<T>(center);
//        }
//        if (i == 4){
//            posx[i*incPosx] = static_cast<T>(center);
//            posy[i*incPosy] = static_cast<T>(center - offset_center);
//            posz[i*incPosz] = static_cast<T>(center);
//        }
//        if (i == 5){
//            posx[i*incPosx] = static_cast<T>(center);
//            posy[i*incPosy] = static_cast<T>(center);
//            posz[i*incPosz] = static_cast<T>(center - offset_center);
//        }
	}
}
	
} /* namespace sim */

# endif
