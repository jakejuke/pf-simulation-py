#include <exec/config3d.hpp>
#include <src/mesh3d.hpp>
#include <src/sim3d.hpp>
#include <src/aux.hpp>
#include <cstdio>
#include <cstddef>

// Packages added by Jules
#include <vector>
// I would love to read/write directly to h5 files
// Could use this library (and install parallel version)
// #include "H5Cpp.h"

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 400
#endif

// num of iterations after which output is generated
// _ _ _ NO LONGER USED _ _ _
#ifndef STEP_SIZE
#define STEP_SIZE 0
#endif

#ifndef OVERWRITE
#define OVERWRITE true
#endif

// Function to check if the loop index matches a user-defined save interval
bool is_save_interval(int index, const std::vector<int>& intervals) {
    for (int interval : intervals) {
        if (index == interval) {
            return true;
        }
    }
    return false;
}

int main() {
	
	// configure output directory name
	const char outdirname[128] = "/Users/jules/Downloads/foo/pfsim_2024-10-17_09h47m09s/";

	// define system size
	const std::size_t m = 32;
	const std::size_t n = 32;
	const std::size_t k = 32;
	const std::size_t localnopmax = LOCAL_NOP_MAX;

	// user-defined save intervals
	std::vector<int> save_intervals = {10, 30, 80, 150};

	std::size_t save_diff = 0;
	std::size_t save_previous = 0;
	std::size_t save_current = 0;
	
	// initialize test case
	mesh::thd::Mesh3d<unsigned short> nopA(m,n,k);
	mesh::thd::Mesh3d<float> valA(m,n,k,localnopmax);
	mesh::thd::Mesh3d<unsigned short> idA(m,n,k,localnopmax);
	
	// create the meshes that we propagate to
	mesh::thd::Mesh3d<unsigned short> nopB(m,n,k);
	mesh::thd::Mesh3d<float> valB(m,n,k,localnopmax);
	mesh::thd::Mesh3d<unsigned short> idB(m,n,k,localnopmax);
	
	// create meshes to store maximum of each op
	// mesh::thd::Mesh2d<float> valMax(m,n);
	mesh::thd::Mesh3d<unsigned short> idMax(m,n,k);
	
	// read from file to init mesh (start config)
	char infilename[128] = "/Users/jules/Downloads/foo/pfsim_2024-10-17_09h47m09s/map.ops";
	CHECK_FILE_READ(mesh::thd::init_from_file, infilename, nopA, valA, idA);
	
	// generate object to manage misorientations
	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	// load misorientation data
	CHECK_FILE_READ(aux::load_array, "./misorientation", misorientation.length, misorientation.data);
	
	// set up buffer to read coefficient maps
	const std::size_t angleres = 6800;
	const std::ptrdiff_t inccoeff = 2;
	aux::Buffer<float> coeffmapsbuffer(angleres*2);

	// files for L and gamma (CHANGE HERE)
	CHECK_FILE_READ(sim::readcoeffmaps, "/Users/jules/Downloads/foo/pfsim_2024-10-17_09h47m09s/L.txt", coeffmapsbuffer.data, inccoeff);
	// now repeat for gamma
	CHECK_FILE_READ(sim::readcoeffmaps, "/Users/jules/Downloads/foo/pfsim_2024-10-17_09h47m09s/gamma.txt", coeffmapsbuffer.data+1, inccoeff);
	
	// copy file with orientation matrices to target directory
	char outfilename[128];
	std::sprintf(outfilename, "%sorimap", outdirname);
	CHECK_FILE_COPY(aux::copy_file, "./orimap", outfilename);
	
	// Write the original dist
	//std::sprintf(outfilename, "%spartition_step000", outdirname);
	// call function to store
	// CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, nopA, valA, idA, OVERWRITE);
	
	// write original dist local max
	std::sprintf(outfilename, "%slm_partition_step000", outdirname);
	// call function to store
	mesh::thd::maxop(nopA, valA, idA, idMax);
	CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, idMax, OVERWRITE);
	
	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	const float deltat = DELTAT;
	// loop for the number of steps
	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		auto status = sim::thd::step(nopA, valA, idA, 
				nopB, valB, idB,
				deltat,
				misorientation.data,
				coeffmapsbuffer.data, inccoeff);
		if (status != 0) {
			std::printf("step function error %d, aborting\n",status);
			return (2);
		}
		
		// save results every "STEP_SIZE" steps
		if (is_save_interval(l, save_intervals)) {

			save_current = l;
			save_diff = save_current - save_previous;
			
			// output time elapsed for calculation
			std::printf("time elapsed for %zu steps: %lfs\n",save_diff,wt.toc());
			
			// genarate filename with number of
			// propagation steps contained
			std::sprintf(outfilename, "%spartition_step%03ld", outdirname, l);

			//call function to store results
			//wt.tic();
			//CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, nopB, valB, idB, OVERWRITE);
			//std::printf("in %fs\n", wt.toc());
			
			// output of local max
			mesh::thd::maxop(nopB, valB, idB, idMax);
			
			// generate filename
			std::sprintf(outfilename, "%slm_partition_step%03ld", outdirname, l);
			//call function to store results
			wt.tic();
			CHECK_FILE_WRITE(mesh::thd::storemesh, outfilename, idMax, OVERWRITE);
			// and check for success of writing
			std::printf("in %fs\n", wt.toc());
			
			save_previous = save_current;

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

	std::printf("terminated successfully\n");
}


////////////////////////////
// Nothing past this line //
////////////////////////////



