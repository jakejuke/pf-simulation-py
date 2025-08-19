#ifndef SRC_SIM_GENPARTICLEFIELD3D_HPP
#define SRC_SIM_GENPARTICLEFIELD3D_HPP 1

#include <src/mesh/mesh3d.hpp>
#include <src/aux/ringproject.hpp>
#include <cstdio>
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
template <typename T, typename R, typename S>
void gen_particle_field3d( std::size_t m, std::size_t n, std::size_t k,
					std::size_t nof_particles, 
					R* posx, std::ptrdiff_t incPosx,
					R* posy, std::ptrdiff_t incPosy,
					R* posz, std::ptrdiff_t incPosz,
					S particle_radius,
					T* field, 
					std::ptrdiff_t incRowField, std::ptrdiff_t incColField, std::ptrdiff_t incLayField,
					T epsilon) {
	
	// set field to zero
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<k; ++l) {
				field[i*incRowField+j*incColField+l*incLayField] = 0;
			}
		}
	}
	
	// loop over particles
	for (std::size_t p=0; p<nof_particles; ++p) {
		auto currentx = posx[p*incPosx];
		auto currenty = posy[p*incPosy];
		auto currentz = posz[p*incPosz];
		
		// consider small environment around each 
		// particles pos
		int istart = static_cast<int>(currentx-particle_radius);
		int iend = static_cast<int>(currentx+particle_radius);
		int jstart = static_cast<int>(currenty-particle_radius);
		int jend = static_cast<int>(currenty+particle_radius);
		int lstart = static_cast<int>(currentz-particle_radius);
		int lend = static_cast<int>(currentz+particle_radius);
		
		// now run over this environment
		for (int i=istart; i<=iend; ++i) {
			for (int j=jstart; j<=jend; ++j) {
				for (int l=lstart; l<=lend; ++l) {
				
					// if distance to particle center is 
					// smaller than radius, set fiel variable to 1;
					// take into account periodic boundary conditions
					// via the projection of indices onto the ring
					S dist_to_center2 = static_cast<S>((i-currentx)*(i-currentx) 
										+ (j-currenty)*(j-currenty) 
										+ (l-currentz)*(l-currentz));
					if (dist_to_center2<=particle_radius*particle_radius) {
						field[aux::ringproject(i,m)*incRowField
								+ aux::ringproject(j,n)*incColField
								+ aux::ringproject(l,k)*incLayField] = epsilon; 
					}
				}
			}
		}
	}
}

// Wrapper function to be able to pass 
// the mesh object only
template <typename T, typename R, typename S>
void gen_particle_field3d(std::size_t nof_particles, 
					R* posx, std::ptrdiff_t incPosx,
					R* posy, std::ptrdiff_t incPosy,
					R* posz, std::ptrdiff_t incPosz,
					S particle_radius,
					mesh::thd::Mesh3d<T>& field,
					T epsilon = 1.) {
	
	
	sim::gen_particle_field3d(field.xsize(), field.ysize(), field.zsize(),
						nof_particles,
						posx, incPosx,
						posy, incPosy,
						posz, incPosz,
						particle_radius,
						field.ptr(), field.xinc(), field.yinc(), field.zinc(),
						epsilon);
	
}

	
} /* namespace sim */

# endif
