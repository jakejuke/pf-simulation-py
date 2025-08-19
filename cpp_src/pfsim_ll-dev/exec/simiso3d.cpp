#include <exec/config3d.hpp>
#include <src/mesh3d.hpp>
#include <src/sim3d.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/walltime.hpp>
#include <src/aux/checkfileop.hpp>
#include <src/aux/copyfile.hpp>
#include <cstdio>
#include <cstddef>

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 10000
#endif

// num of iterations after which output is generated
#ifndef STEP_SIZE
#define STEP_SIZE 100
#endif

#ifndef OVERWRITE
#define OVERWRITE false
#endif


int main() {
	
	// configure output directory name
	const char outdirname[32] = "./../workdir/3D_test_2/";

	// define system size
	const std::size_t m = 120;
	const std::size_t n = 120;
	const std::size_t k = 120;
	const std::size_t localnopmax = LOCAL_NOP_MAX;
	
	// initialize test case
	mesh::thd::Mesh3d<unsigned short> nopA(m,n,k);
	mesh::thd::Mesh3d<float> valA(m,n,k,localnopmax);
	mesh::thd::Mesh3d<unsigned short> idA(m,n,k,localnopmax);
	
	// create the meshes that we propagate to
	mesh::thd::Mesh3d<unsigned short> nopB(m,n,k);
	mesh::thd::Mesh3d<float> valB(m,n,k,localnopmax);
	mesh::thd::Mesh3d<unsigned short> idB(m,n,k,localnopmax);
	
	// create meshes to store maximum of each op
	mesh::thd::Mesh3d<float> valMax(m,n,k);
	mesh::thd::Mesh3d<unsigned short> idMax(m,n,k);
	
	// read from file
	char filename[32] = "./../data/120erDataSet/ops";
	CHECK_FILE_READ(mesh::thd::init_from_file, filename, nopA, valA, idA);
	
	// copy file with orientation matrices to target directory
	char outfilename[128];
	std::sprintf(outfilename, "%sorimap", outdirname);
	CHECK_FILE_COPY(aux::copy_file, "./orimap", outfilename);
	
	// Write the original dist
	//std::sprintf(outfilename, "%spartition_step000", outdirname);
	// call function to store
	//CHECK_FILE_READ(mesh::thd::storemesh, outfilename, nopA, valA, idA, OVERWRITE);
	
	std::sprintf(outfilename, "%slm_partition_step000", outdirname);
	// call function to store
	mesh::thd::maxop(nopA, valA, idA, idMax);
	CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, idMax, OVERWRITE);
	
	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	const float deltat = DELTAT;
	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		auto status = sim::thd::stepiso(nopA, valA, idA, 
				nopB, valB, idB,
				deltat
   									);
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
			// wt.tic();
			// CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, nopB, valB, idB, OVERWRITE);
			// std::printf("in %fs\n", wt.toc());
			
			// output of local max
			mesh::thd::maxop(nopB, valB, idB, idMax);
			
			// generate filename
			std::sprintf(outfilename, "%slm_partition_step%03ld", outdirname, l);
			
			//call function to store results
			wt.tic();
			CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, idMax, OVERWRITE);
			std::printf("in %fs\n", wt.toc());
			
			// reset start timer
			wt.tic();
		}
		

		// swap the pointers to propagate in the
		// other direction now
		mesh::swap(nopA, nopB);
		mesh::swap(valA, valB);
		mesh::swap(idA, idB);
	}
	
	// output final distribution
	std::sprintf(outfilename, "%spartition_step%03ld", outdirname, static_cast<std::size_t>(NUM_ITER));
	CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, nopA, valA, idA, OVERWRITE);

	std::printf("exit with success\n");
}
