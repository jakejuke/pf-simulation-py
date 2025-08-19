#include <exec/config2d.hpp>
#include <cstdio>
#include <cstddef>
#include <src/sim/genparticlepos2d.hpp>
#include <src/sim/genparticlefield2d.hpp>
#include <src/sim/storeparticlepos2d.hpp>
#include <src/mesh/mesh2d.hpp>
#include <src/mesh/storemesh2d.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/checkfileop.hpp>

#ifndef OVERWRITE
#define OVERWRITE true
#endif

int main() {

	// define system size
	std::size_t numparticles = NOF_PARTICLES;
	std::size_t m = 300;
	std::size_t n = 300;

	// setup memory space
	mesh::twd::Mesh2d<float> particlefield(m, n);
	aux::Buffer<std::size_t> particlepos(numparticles*2);

	// generate particle positions
	sim::gen_particle_pos2d(m,n,
			numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2);

	char filenamepos[] = "../data/particlePositions/pos2d";
	CHECK_FILE_WRITE(sim::store_particle_pos2d,
			filenamepos,
			numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2,
			OVERWRITE);

	// define radius and generate the field
	float radius = static_cast<float>(PARTICLE_RADIUS);
	sim::gen_particle_field2d(numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2,
			radius,
			particlefield,
			(float) 3.);

	char filenamefield[] = "../data/particlePositions/phi2d";
	CHECK_FILE_WRITE(mesh::twd::storemesh,
					 filenamefield, 0, particlefield, true);

}


