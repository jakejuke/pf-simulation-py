#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <src/mesh/storemesh2d.hpp>
#include <src/aux/buffer.hpp>
#include <cstdio>
#include <cstddef>

//number of iterations
#ifndef NUM_ITER
#define NUM_ITER 10 
#endif

#ifndef STEP_SIZE
#define STEP_SIZE 2
#endif


int main() {

	// define system size
	const std::size_t m = 300;
	const std::size_t n = 300;
	
	// initialize test case
	mesh::twd::Mesh2d<unsigned short> nopA(m,n);
	mesh::twd::Mesh2d<float> valA(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idA(m,n,0);
	
	//read from file
	char filename[32] = "./../data/testcase/ops";
	auto linesread = mesh::twd::init_from_file(filename, nopA, valA, idA);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error, aborting\n");
		return (2);
	}

	
	// write mesh to file
	char outfilename [32] = "./ops_output.out";
	auto lineswritten = mesh::twd::storemesh(outfilename, nopA, valA, idA, true);
	std::printf("lines written: %d\n", lineswritten);


}
