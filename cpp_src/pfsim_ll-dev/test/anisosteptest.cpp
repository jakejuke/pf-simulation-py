#include <src/mesh2d.hpp>
#include <src/sim2d.hpp>
#include <src/aux.hpp>
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
	
	// generate object to manage misorientations
	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	// load misorientation data
	auto elementsread = aux::load_array("../exec/misorientation", NOF_GRAINS, misorientation.data);
	// check for success
	if (elementsread==-1 || elementsread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("number of elements read for misorientation: %d\n", elementsread);
	
	// set up buffer to read coefficient maps
	const std::size_t angleres = 6800;
	aux::Buffer<float> coeffmapsbuffer(angleres*2);
	linesread = sim::readcoeffmaps("../data/coeffmaps/L2d.txt", coeffmapsbuffer.data, INCCOEFF);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("lines read for coeffmaps init: %d\n", linesread);
	// now repeat for gamma
	linesread = sim::readcoeffmaps("../data/coeffmaps/gamma2d.txt", coeffmapsbuffer.data+1, INCCOEFF);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("lines read for coeffmapsinit init: %d\n", linesread);
	
	
	// set up buffers needed for propagation
	aux::Buffer<float> valbuf(INCBUFF*NOP_LOCAL_MAX);
	aux::Buffer<unsigned short> idbuf(NOP_LOCAL_MAX);

	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		sim::twd::step(nopA, valA, idA, 
				nopB, valB, idB,
				misorientation.data,
				coeffmapsbuffer.data, INCCOEFF,
				NOP_LOCAL_MAX,
				valbuf.data,
				idbuf.data);
		
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
	

	std::printf("exit with success\n");
}
