#include <src/mesh2d.hpp>
#include <src/sim/step2dcoefftest.hpp>
#include <src/sim2d.hpp>
//#include <src/aux.hpp>
#include <src/aux/lowtri.hpp>
#include <src/aux/savearray.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/walltime.hpp>
#include <src/aux/savearray.hpp>
#include <src/sim/genorientation.hpp>
#include <src/sim/saveorientation.hpp>
#include <src/sim/calcmisorientation.hpp>
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

#ifndef OVERWRITE
#define OVERWRITE true
#endif

#ifndef NOF_GRAINS
#define NOF_GRAINS 656
#endif

#ifndef INCCOEFF
#define INCCOEFF 2
#endif


int main() {
	
	// configure output directory name
	const char outdirname[32] = "./../workdir/anisotestcase/";

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
	// mesh::twd::Mesh2d<float> valMax(m,n);
	mesh::twd::Mesh2d<unsigned short> idMax(m,n);
	
	// read from file to init mesh
	char infilename[] = "./../workdir/anisotestcase/partition_step100";
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
	auto elementsread = aux::load_array("./../exec/misorientation", misorientation.length, misorientation.data);
	// check for success
	if (elementsread==-1 || elementsread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("number of elements read for misorientation: %d\n", elementsread);
	
// 	// create objects to allocate memory
// 	aux::Buffer<float> orientationbuffer(NOF_GRAINS*9);
// 	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
// 	sim::gen_orientation(NOF_GRAINS,
// 						 orientationbuffer);
// 	sim::calc_all_misorientation(NOF_GRAINS,
// 								orientationbuffer,
// 								misorientation);
// 	std::printf("(%d, %d): %f\n", 26, 25, misorientation(25,24));
// 	std::printf("(%d, %d): %f\n", 652, 45, misorientation(651,44));
	
	// set up buffer to read coefficient maps
	const std::size_t angleres = 6800;
	aux::Buffer<float> coeffmapsbuffer(angleres*2);
	linesread = sim::readcoeffmaps("../data/coeffmapsAniso/L2d.txt", coeffmapsbuffer.data, INCCOEFF);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("lines read for coeffmaps init of L: %d\n", linesread);
	// now repeat for gamma
	linesread = sim::readcoeffmaps("../data/coeffmapsAniso/gamma2d.txt", coeffmapsbuffer.data+1, INCCOEFF);
	// check for success
	if (linesread==-1 || linesread==0) {
		std::printf("data file input error: %d, aborting\n",linesread);
		return (2);
	}
	std::printf("lines read for coeffmapsinit init of gamma: %d\n", linesread);
	
	// copy file with orientation matrices to target directory
	char outfilename[128];
	std::sprintf(outfilename, "%sorimap", outdirname);
	auto copyretvalue = aux::copy_file("./../exec/orimap", outfilename);
	if(copyretvalue) {
		std::printf("orimap copied to %s\n", outfilename);
	} else {
		std::printf("Error when trying to copy orimap\n");
	}
	
	// Write the original dist
	std::sprintf(outfilename, "%spartition_step000", outdirname);
	// call function to store
	/*
	auto lineswritten = mesh::twd::storemesh(outfilename, nopA, valA, idA);
	// and check for success of writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d lines written to %s\n",lineswritten, outfilename);
	*/
	
	// write original dist local max
	std::sprintf(outfilename, "%slm_partition_step000", outdirname);
	// call function to store
	mesh::twd::maxop(nopA, valA, idA, idMax);
	/*
	lineswritten = mesh::twd::storemesh(outfilename, idMax);
	// and check for success of writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d lines written to %s\n",lineswritten, outfilename);
	*/
	
	// set up timer
	//aux::WallTime<float> wt;
	//wt.tic();

	// loop for the number of steps
	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		auto status = sim::twd::step(4, 4, valA.opbufsize(),
						(std::size_t) 0, (std::size_t) 0,
						nopA.xsize(), nopA.ysize(),
						nopA.ptr(), nopA.xinc(), nopA.yinc(), 
						valA.ptr(), valA.xinc(), valA.yinc(),
						idA.ptr(), idA.xinc(), idA.yinc(),
						nopB.ptr(), nopB.xinc(), nopB.yinc(),
						valB.ptr(), valB.xinc(), valB.yinc(),
						idB.ptr(), idB.xinc(), idB.yinc(),
						misorientation.data,
						coeffmapsbuffer.data, INCCOEFF);
		
// 		sim::twd::step(nopA, valA, idA, 
// 				nopB, valB, idB,
// 				misorientation.data,
// 				coeffmapsbuffer.data, INCCOEFF);
		if (status != 0) {
			std::printf("step function error %d, aborting\n",status);
			return (2);
		}
		
		// save results every "STEP_SIZE" steps
		if (l%STEP_SIZE == 0) {
			
			// output time elapsed for calculation
			//std::printf("time elapsed for %d steps: %lfs\n",STEP_SIZE,wt.toc());

			// genarate filename with number of
			// propagation steps contained
			std::sprintf(outfilename, "%spartition_step%03ld", outdirname, l);

			//call function to store results
			/*
			wt.tic();
			lineswritten = mesh::twd::storemesh(outfilename, nopB, valB, idB);
			// and check for success of writing
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error: %d, aborting\n", lineswritten);
				return (2);
			}
			std::printf("%d lines written to %s in %fs\n",lineswritten, outfilename, wt.toc());
			*/
			
			// output of local max
			mesh::twd::maxop(nopB, valB, idB, idMax);
			
			// generate filename
			std::sprintf(outfilename, "%slm_partition_step%03ld", outdirname, l);
			
			//call function to store results
			/*
			wt.tic();
			lineswritten = mesh::twd::storemesh(outfilename, idMax);
			// and check for success of writing
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error: %d, aborting\n", lineswritten);
				return (2);
			}
			std::printf("%d lines written to %s in %fs\n",lineswritten, outfilename, wt.toc());
			*/
			
			// reset start timer
			//wt.tic();
		}
		

		// swap the pointers to propagate in the
		// other direction now
		mesh::swap(nopA, nopB);
		mesh::swap(valA, valB);
		mesh::swap(idA, idB);
	}

	std::printf("terminated successfully\n");
}
