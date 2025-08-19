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
	const std::size_t m = 4;
	const std::size_t n = 4;
	
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
	char infilename[32] = "./testfileops4";
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
	auto elementsread = aux::load_array("../exec/misorientation", misorientation.length, misorientation.data);
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


	// do some testing
	for (std::size_t i=0; i<misorientation.length; ++i) {
		misorientation.data[i]++;
	}
	
		
}
