#include <exec/config2d.hpp>
#include <src/mesh2d.hpp>
#include <src/sim2d.hpp>
#include <src/aux.hpp>
#include <cstdio>
#include <cstddef>
#include <src/sim/savecoeffmap.hpp>

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 20000
#endif

// num of iterations after which output is generated
#ifndef STEP_SIZE
#define STEP_SIZE 200
#endif

#ifndef OVERWRITE
#define OVERWRITE true
#endif


int main() {
	
	// configure output directory name
	const char outdirname[128] = "/tmp/pfsim_89x4ez4j/";

	// define system size
	const std::size_t m = 128;
	const std::size_t n = 128;
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
	
	// read from file to init mesh (start config)
	char infilename[128] = "/tmp/pfsim_89x4ez4j/map.ops";
	CHECK_FILE_READ(mesh::twd::init_from_file, infilename, nopA, valA, idA);
	
	// generate object to manage misorientations
	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	// load misorientation data
	CHECK_FILE_READ(aux::load_array, "./misorientation", misorientation.length, misorientation.data);
	
	const std::ptrdiff_t inccoeff = 2;
	// set up buffer to read coefficient maps
	const std::size_t angleres = 6800;
	aux::Buffer<float> coeffmapsbuffer(angleres*2);

	// files for L and gamma (CHANGE HERE)
	CHECK_FILE_READ(sim::readcoeffmaps, "/tmp/pfsim_89x4ez4j/L.txt", coeffmapsbuffer.data, inccoeff);
	// now repeat for gamma
	CHECK_FILE_READ(sim::readcoeffmaps, "/tmp/pfsim_89x4ez4j/gamma.txt", coeffmapsbuffer.data+1, inccoeff);
	

	
	
	//read out misori,L for debugging
	/*std::size_t y = 2;
	float valbuf[y] = {1, 1};
	std::size_t idbuf[y] = {230,231};

	std::printf("ID1 %f\n",static_cast<double>(idbuf[0]));
	std::printf("ID2 %f\n",static_cast<double>(idbuf[1]));


	std::printf("%f ",misorientation.data[(idbuf[0]-GRAIN_ID_START)
			*(idbuf[0]-1-GRAIN_ID_START)/2
			+idbuf[1]-GRAIN_ID_START]*100-1);

	std::printf("\n");


	float Lcoeff = sim::get_Lcoeff(y,
			misorientation.data,
			valbuf, 1,
			idbuf, 1,
			coeffmapsbuffer.data, 2);

	std::printf("result for L: %f\n", Lcoeff);


	float index = static_cast<std::size_t>(misorientation.data[(idbuf[0]-GRAIN_ID_START)
			*(idbuf[0]-1-GRAIN_ID_START)/2
			+idbuf[1]-GRAIN_ID_START]*100-1);

	std::printf("result for index: %f\n", index); */
	

	/*
	//Check what is read into coeffmapsbuffer:
	auto lineswritten = sim::save_coeffmap("./readLmap", angleres, coeffmapsbuffer.data);
	// check for successful writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d lines written to %s\n",lineswritten, "./readLmap");
	*/



	// copy file with orientation matrices to target directory
	char outfilename[192];
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
		auto status = sim::twd::step(nopA, valA, idA, 
				nopB, valB, idB,
				deltat,
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
			// and check for success of writing
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

