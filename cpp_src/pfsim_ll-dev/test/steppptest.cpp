#include <src/mesh2d.hpp>
#include <src/sim2d.hpp>
#include <src/aux.hpp>
#include <src/sim/genparticlepos2d.hpp>
#include <src/sim/genparticlefield2d.hpp>
#include <src/sim/stepisopp2d.hpp>
#include <cstdio>
#include <cstddef>

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 1
#endif

// num of iterations after which output is generated
#ifndef STEP_SIZE
#define STEP_SIZE 1
#endif

#ifndef NOF_GRAINS
#define NOF_GRAINS 656
#endif

#ifndef INCCOEFF
#define INCCOEFF 2
#endif


int main() {

	// define system size
	const std::size_t m = 300;
	const std::size_t n = 300;
	
	// initialize test case
	mesh::twd::Mesh2d<unsigned short> nopA(m,n);
	mesh::twd::Mesh2d<float> valA(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idA(m,n,0);
	
	// create the meshes that we propagate to
	mesh::twd::Mesh2d<unsigned short> nopB(m,n);
	mesh::twd::Mesh2d<float> valB(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idB(m,n,0);
	
	// create meshes to store maximum of each op
	mesh::twd::Mesh2d<float> valMax(m,n);
	mesh::twd::Mesh2d<unsigned short> idMax(m,n);
	
	// read from file to init mesh
	char infilename[32] = "../data/testcase/ops";
	auto linesread = mesh::twd::init_from_file(infilename, nopA, valA, idA);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("lines read for mesh init: %d\n", linesread);
	
	// setup pinned particles
	std::size_t numparticles = 63;
	aux::Buffer<std::size_t> particlepos(numparticles*2);
	mesh::twd::Mesh2d<float> particlefield(m, n);
	
	// generate particle positions
	sim::gen_particle_pos2d(m,n,
			numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2);
	// generate field out of these positions
	float radius = 3.0;
	float epsilon = 1.0;
	sim::gen_particle_field2d(numparticles,
			particlepos.data, 2,
			particlepos.data+1, 2,
            radius,
			particlefield,
			epsilon);
	
	
	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		sim::twd::stepisopp(nopA, valA, idA, 
				nopB, valB, idB,
				particlefield);
		
		// save results
		/*
		if (l%STEP_SIZE == 0) {
			
			// output time elapsed for calculation
			std::printf("time elapsed for %d steps: %lfs\n",STEP_SIZE,wt.toc());

			// genarate filename with number of
			// propagation steps contained
			char outfilename[64];
			std::sprintf(outfilename, "./../workdir/anisotestcase/aniso_%03ld", l);

			//call function to store results
			auto lineswritten = mesh::twd::storemesh(outfilename, nopB, valB, idB);
			// and check for success of writing
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error: %d, aborting\n", lineswritten);
				return (2);
			}
			std::printf("%d lines written to %s\n",lineswritten, outfilename);
			
			// output of local max
			mesh::twd::maxop(nopB, valB, idB, valMax, idMax);
			
			// generate filename
			std::sprintf(outfilename, "./../workdir/anisotestcase/lm_aniso_%03ld", l);
			
			//call function to store results
			lineswritten = mesh::twd::storemesh(outfilename, valMax, idMax);
			// and check for success of writing
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error: %d, aborting\n", lineswritten);
				return (2);
			}
			std::printf("%d lines written to %s\n",lineswritten, outfilename);
			
			// reset start timer
			wt.tic();
		}
		*/

		// swap the pointers to propagate in the
		// other direction now
		mesh::swap(nopA, nopB);
		mesh::swap(valA, valB);
		mesh::swap(idA, idB);
	}
	
	std::printf("some values:\n");
	for (std::size_t i=0; i<10; ++i) {
		for (std::size_t j=0; j<10; ++j) {
			std::printf("%f ", valA(i,j,0));
		}
		std::printf("\n");
	}

	std::printf("exit with success\n");
}
