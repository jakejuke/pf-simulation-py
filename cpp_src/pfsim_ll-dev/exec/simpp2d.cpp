#include <exec/config2d.hpp>
#include <src/mesh2d.hpp>
#include <src/sim2d.hpp>
#include <src/aux.hpp>
#include <cstdio>
#include <cstddef>

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 100000
#endif

// num of iterations after which output is generated
#ifndef STEP_SIZE
#define STEP_SIZE 100
#endif

#ifndef OVERWRITE
#define OVERWRITE true
#endif


int main() {
	
	// configure output directory name
	const char outdirname[64] = "./../workdir/2DCircle4Particles/";

	// define system size
	const std::size_t m = 240;
	const std::size_t n = 240;
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
	// mesh::twd::Mesh2d<float> valMax(m,n);
	mesh::twd::Mesh2d<unsigned short> idMax(m,n);
	
	// create mesh to store particle field
	mesh::twd::Mesh2d<float> field(m,n);
	
	// read from file to init mesh
	char infilename[64] = "./../data/2DCircle/ops";
	CHECK_FILE_READ(mesh::twd::init_from_file, infilename, nopA, valA, idA);
	
	// generate object to manage misorientations
	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	// load misorientation data
	CHECK_FILE_READ(aux::load_array, "./misorientation", misorientation.length, misorientation.data);
	
	// set up buffer to read coefficient maps
	const std::size_t angleres = 6800;
	const std::ptrdiff_t inccoeff = 2;
	aux::Buffer<float> coeffmapsbuffer(angleres*2);
	CHECK_FILE_READ(sim::readcoeffmaps, "../data/coeffmapsAniso/L_2p8.txt", coeffmapsbuffer.data, inccoeff);
	// now repeat for gamma
	CHECK_FILE_READ(sim::readcoeffmaps, "../data/coeffmapsAniso/gamma_2p8.txt", coeffmapsbuffer.data+1, inccoeff);
	
	// read particle field from file
	std::sprintf(infilename, "./../data/particlePositions/phi2d");
	float epsilon = static_cast<float>(FIELD_EPSILON);
	CHECK_FILE_READ(mesh::twd::init_from_file, infilename, field, epsilon);
	
	// copy file with orientation matrices to target directory
	char outfilename[128];
	std::sprintf(outfilename, "%sorimap", outdirname);
	CHECK_FILE_COPY(aux::copy_file, "./orimap", outfilename);
	
	// Write the original dist
	std::sprintf(outfilename, "%spartition_step000", outdirname);
	// call function to store
	CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, nopA, valA, idA, OVERWRITE);
	
	// write original dist local max
	std::sprintf(outfilename, "%slm_partition_step000", outdirname);
	// call function to store
	mesh::twd::maxop(nopA, valA, idA, idMax);
	CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, idMax, OVERWRITE);
	
	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	const float deltat = DELTAT;
	// loop for the number of steps
	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		auto status = sim::twd::steppp(nopA, valA, idA, 
				nopB, valB, idB,
				deltat,
				field,
				misorientation.data,
				coeffmapsbuffer.data, inccoeff);
		if (status != 0) {
			std::printf("step function error %d, aborting\n",status);
			return (2);
		}
		
		// save results every "STEP_SIZE" steps
		if (l%STEP_SIZE == 0) {
			
			// output time elapsed for calculation
			std::printf("time elapsed for %d steps: %lfs\n",STEP_SIZE,wt.toc());

			// genarate filename with number of
			// propagation steps contained
			std::sprintf(outfilename, "%spartition_step%03ld", outdirname, l);

			//call function to store results
			wt.tic();
			CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, nopB, valB, idB, OVERWRITE);
			std::printf("in %fs\n", wt.toc());
			
			// output of local max
			mesh::twd::maxop(nopB, valB, idB, idMax);
			
			// generate filename
			std::sprintf(outfilename, "%slm_partition_step%03ld", outdirname, l);
			
			//call function to store results
			wt.tic();
			CHECK_FILE_WRITE(mesh::twd::storemesh, outfilename, idMax, OVERWRITE);
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

	std::printf("terminated successfully\n");
}
