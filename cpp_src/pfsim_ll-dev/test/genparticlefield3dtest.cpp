#include <exec/config3d.hpp>
#include <cstdio>
#include <cstddef>
#include <iostream>
#include <src/sim/genparticlepos3d.hpp>
#include <src/sim/genparticlefield3d.hpp>
#include <src/sim/storeparticlepos3d.hpp>
#include <src/mesh/mesh3d.hpp>
#include <src/mesh/storemesh3d.hpp>
#include <src/mesh/init3d.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/checkfileop.hpp>

#ifndef OVERWRITE
#define OVERWRITE true
#endif

#ifndef NOF_PARTICLES
#define NOF_PARTICLES 63
#endif

#ifndef PARTICLE_RADIUS
#define PARTICLE_RADIUS 3
#endif

#ifndef FIELD_EPSILON
#define FIELD_EPSILON 1.0
#endif

int main() {

	// define system size
	std::size_t numparticles = NOF_PARTICLES;
	std::size_t m = 120;
	std::size_t n = 120;
	std::size_t k = 120;

	// setup memory space
	mesh::thd::Mesh3d<float> particlefield(m, n, k);
	aux::Buffer<std::size_t> particlepos(numparticles*3);

	// generate particle positions
	sim::gen_particle_pos3d(m,n,k,
			numparticles,
			particlepos.data, 3,
			particlepos.data+1, 3,
			particlepos.data+2, 3);
	
	for (std::size_t i=0; i<15; ++i) {
		for (std::size_t j=0; j<3; ++j) {
			std::cout << particlepos.data[3*i+j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "particle pos end" << std::endl;

	char filenamepos[] = "../data/particlePositions/pos3d";
// 	CHECK_FILE_WRITE(sim::store_particle_pos3d,
// 			filenamepos,
// 			numparticles,
// 			particlepos.data, 3,
// 			particlepos.data+1, 3,
// 			particlepos.data+2, 3,
// 			OVERWRITE);
	// define radius and generate the field
	float radius = static_cast<float>(PARTICLE_RADIUS);
	float epsilon = static_cast<float>(FIELD_EPSILON);
	sim::gen_particle_field3d(numparticles,
			particlepos.data, 3,
			particlepos.data+1, 3,
			particlepos.data+2, 3,
			radius,
			particlefield,
			epsilon);

	char filenamefield[] = "./phi3d";
	CHECK_FILE_WRITE(mesh::thd::storemesh,
					 filenamefield, 0, particlefield, true);
	std::size_t izero = 84;
	std::size_t jzero = 40;
	std::size_t lzero = 57;
	for (std::size_t i=izero; i<izero+35; ++i) {
		for (std::size_t j=jzero; j<jzero+35; ++j) {
			std::cout << particlefield(i,j,lzero) << " ";
		}
		std::cout << std::endl;
	}
	
	mesh::thd::Mesh3d<float> loadedfield(m,n,k);
	CHECK_FILE_READ(mesh::thd::init_from_file, filenamefield, loadedfield, epsilon);
	
	std::cout << "loaded field:" << std::endl;
	for (std::size_t i=izero; i<izero+35; ++i) {
		for (std::size_t j=jzero; j<jzero+35; ++j) {
			std::cout << loadedfield(i,j,lzero) << " ";
		}
		std::cout << std::endl;
	}
	
}


