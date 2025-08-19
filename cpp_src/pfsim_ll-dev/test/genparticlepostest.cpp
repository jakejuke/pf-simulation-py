#include <iostream>
#include <src/sim/genparticlepos.hpp>
#include <cstddef>
#include <src/mesh/mesh2d.hpp>
#include <src/aux/buffer.hpp>

int main() {

	std::size_t numparticles = 5;
	std::size_t m = 30;
	std::size_t n = 30;

	mesh::twd::Mesh2d<float> particlefield(m, n);
	aux::Buffer<std::size_t> particlepos(numparticles*2);

	sim::gen_particle_pos(m,n,
			numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2);

	for (std::size_t i=0; i<numparticles; ++i) {
		std::cout << particlepos[2*i] << " " << particlepos[2*i+1] << std::endl;
	}

}


