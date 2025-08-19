#include <exec/config2d.hpp>
#include <src/mesh2d.hpp>
#include <src/sim2d.hpp>
#include <src/aux.hpp>
#include <cstdio>
#include <cstddef>

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 20000
#endif

// num of iterations after which output is generated
#ifndef STEP_SIZE
#define STEP_SIZE 10000
#endif

#ifndef OVERWRITE
#define OVERWRITE false
#endif

int main() {
	
	// configure output directory name
	const char outdirname[32] = "./../wdir/Aniso_120_2D_1/";

	// define system size
	const std::size_t m = 300;
	const std::size_t n = 300;
	const std::size_t localnopmax = LOCAL_NOP_MAX;
	
	// initialize test case
	mesh::twd::Mesh2d<unsigned short> nopA(m,n);
	mesh::twd::Mesh2d<float> valA(m,n,localnopmax);
	mesh::twd::Mesh2d<unsigned short> idA(m,n,localnopmax);
	
	// create the meshes that we propagate to
	mesh::twd::Mesh2d<unsigned short> nopB(m,n);
	mesh::twd::Mesh2d<float> valB(m,n,localnopmax);
	mesh::twd::Mesh2d<unsigned short> idB(m,n,localnopmax);
	
	// create meshes to store maximum of each op
	mesh::twd::Mesh2d<float> valMax(m,n);
	mesh::twd::Mesh2d<unsigned short> idMax(m,n);
	
	// read from file
	char filename[57] = "../data/300x300_Aniso/ops_300_2D_Aniso";
	CHECK_FILE_READ(mesh::twd::init_from_file, filename, nopA, valA, idA);
	
	// Write the original dist
	char outfilename[128];
	std::sprintf(outfilename, "%spartition_step000", outdirname);
	// call function to store
	CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, nopA, valA, idA, OVERWRITE)
	
	std::sprintf(outfilename, "%slm_partition_step000", outdirname);
	// call function to store
	mesh::twd::maxop(nopA, valA, idA, idMax);
	CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, idMax, OVERWRITE);
	

	
	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	const float deltat = DELTAT;
	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		auto status = sim::twd::stepiso(nopA, valA, idA, 
				nopB, valB, idB,
				deltat);
		if (status != 0) {
			std::printf("step function error %d, aborting\n",status);
			return (2);
		}
		
		// save results
		if (l%STEP_SIZE == 0) {
			
			// output time elapsed for calculation
			std::printf("time elapsed for %d steps: %lfs\n",STEP_SIZE,wt.toc());

			// genarate filename with number of
			// propagation steps contained
			std::sprintf(outfilename, "%spartition_step%03ld", outdirname, l);

			//call function to store results
			wt.tic();
			CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, nopB, valB, idB, OVERWRITE);
			std::printf("took %fs\n", wt.toc());
			
			// output of local max
			mesh::twd::maxop(nopB, valB, idB, idMax);
			
			// generate filename
			std::sprintf(outfilename, "%slm_partition_step%03ld", outdirname, l);
			
			//call function to store results
			wt.tic();
			CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, idMax, OVERWRITE);
			std::printf("took %fs\n", wt.toc());
			
			// reset start timer
			wt.tic();
		}
		

		// swap the pointers to propagate in the
		// other direction now
		mesh::swap(nopA, nopB);
		mesh::swap(valA, valB);
		mesh::swap(idA, idB);
	}

	std::printf("terminated successfully\n");
}
