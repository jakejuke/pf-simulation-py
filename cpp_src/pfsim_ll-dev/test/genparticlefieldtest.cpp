#include <iostream>
#include <cstdio>
#include <src/sim/genparticlepos2d.hpp>
#include <src/sim/genparticlefield2d.hpp>
#include <src/sim/storeparticlepos2d.hpp>
#include <cstddef>
#include <src/mesh/mesh2d.hpp>
#include <src/mesh/storemesh2d.hpp>
#include <src/aux/buffer.hpp>

int main() {

	std::size_t numparticles = 9;
	std::size_t m = 40;
	std::size_t n = 40;

	mesh::twd::Mesh2d<float> particlefield(m, n);
	aux::Buffer<std::size_t> particlepos(numparticles*2);

	sim::gen_particle_pos2d(m,n,
			numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2);

	for (std::size_t i=0; i<numparticles; ++i) {
		std::cout << particlepos[2*i] << " " << particlepos[2*i+1] << std::endl;
	}

	char filenamepos[] = "particle_positions.txt";
	sim::store_particle_pos2d(filenamepos,
			numparticles,
			particlepos.data, 2,
                        particlepos.data+1, 2,
			true);

	float radius = 5.5;
	/*
	sim::gen_particle_field(m,n,
			numparticles,
			particlepos.data, 2, 
			particlepos.data+1, 2,
		        radius,
			particlefield.ptr(), particlefield.xinc(), particlefield.yinc());	    */
	sim::gen_particle_field2d(numparticles,
			particlepos.data, 2,
                        particlepos.data+1, 2,
                        radius,
			particlefield,
			(float) 3.);

	for (std::size_t i=0; i<particlefield.xsize(); ++i) {
		for (std::size_t j=0; j<particlefield.ysize(); ++j) {
			std::cout <<  particlefield(i,j) << " ";
		}
		std::cout << std::endl;
	}

	char filenamefield[] = "particlefield.txt";
	mesh::twd::storemesh(filenamefield, particlefield, 0, true);

}


