#include <exec/config3d.hpp>
#include <cstdio>
#include <cstddef>
#include <src/sim/genparticlepos3d.hpp>
#include <src/sim/genparticlefield3d.hpp>
#include <src/sim/storeparticlepos3d.hpp>
#include <src/mesh/mesh3d.hpp>
#include <src/mesh/storemesh3d.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/checkfileop.hpp>

#ifndef OVERWRITE
#define OVERWRITE true
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

	char filenamepos[] = "../data/particlePositions/pos3d";
	CHECK_FILE_WRITE(sim::store_particle_pos3d,
			filenamepos,
			numparticles,
			particlepos.data, 3,
			particlepos.data+1, 3,
			particlepos.data+2, 3,
			OVERWRITE);
	// define radius and generate the field
	float radius = static_cast<float>(PARTICLE_RADIUS);
	sim::gen_particle_field3d(numparticles,
			particlepos.data, 3,
			particlepos.data+1, 3,
			particlepos.data+2, 3,
			radius,
			particlefield,
			(float) 3.);

	char filenamefield[] = "../data/particlePositions/phi3d";
	CHECK_FILE_WRITE(mesh::thd::storemesh,
					 filenamefield, 0, particlefield, true);
}


