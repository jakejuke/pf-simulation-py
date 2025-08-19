#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <src/mesh/storemesh2d.hpp>
#include <src/mesh/swap.hpp>
#include <src/sim/stepiso2d.hpp>
#include <src/aux/buffer.hpp>
#include <cstdio>
#include <cstddef>

//number of iterations
#ifndef NUM_ITER
#define NUM_ITER 2 
#endif

#ifndef STEP_SIZE
#define STEP_SIZE 1
#endif


int main() {

	// define system size
	const std::size_t m = 4;
	const std::size_t n = 4;
	
	// initialize test case
	mesh::twd::Mesh2d<unsigned short> nopA(m,n);
	mesh::twd::Mesh2d<float> valA(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idA(m,n,0);
	
	//create the meshes that we propagate to
	mesh::twd::Mesh2d<unsigned short> nopB(m,n);
	mesh::twd::Mesh2d<float> valB(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idB(m,n,0);
	
	//read from file
	char filename[32] = "./testfileops4";
	auto linesread = mesh::twd::init_from_file(filename, nopA, valA, idA);
	std::printf("lines read: %d\n", linesread);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error, aborting\n");
		return (2);
	}

	
	// set up buffers needed for propagation
	aux::Buffer<float> valbuf(INCBUFF*NOP_LOCAL_MAX);
	aux::Buffer<unsigned short> idbuf(NOP_LOCAL_MAX);

	
	for (std::size_t l=0; l<NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		sim::twd::stepiso(nopA, valA, idA, 
						nopB, valB, idB
						);
		

		// save results
		if (l!=0 && l%STEP_SIZE == 0) {
			char outfilename[64];
			std::sprintf(outfilename, "./dump/ops_%ld.out", l);
			auto lineswritten = mesh::twd::storemesh(outfilename, nopB, valB, idB);
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error, aborting\n");
				return (2);
			}
			std::printf("%d lines written to %s\n",lineswritten, outfilename);
		}

		// swap the pointers to propagate in the
		// other direction now
		mesh::swap(nopA, nopB);
		mesh::swap(valA, valB);
		mesh::swap(idA, idB);
	}
	
}
