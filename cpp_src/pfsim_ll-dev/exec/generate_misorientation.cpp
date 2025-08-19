#include <exec/config3d.hpp>
#include <iostream>
#include <cstddef>
#include <src/aux/lowtri.hpp>
#include <src/aux/savearray.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/checkfileop.hpp>
#include <src/aux/walltime.hpp>
#include <src/aux/savearray.hpp>
#include <src/aux/loadarray.hpp>
#include <src/aux/loadori.hpp>
#include <src/sim/genorientation.hpp>
#include <src/sim/saveorientation.hpp>
#include <src/sim/calcmisorientation.hpp>

int main() {
	
	// create objects to allocate memory
	aux::Buffer<float> orientationbuffer(NOF_GRAINS*9);
	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	
	// create timer objects
	aux::WallTime<float> wt;
	
	/*
	// generate orientations (randomly, comment out for own orientations)
	wt.tic();
	sim::gen_orientation(NOF_GRAINS,
						 orientationbuffer);
	std::printf("time elapsed for generation: %fs\n", wt.toc());
	

	// write misorientation maps out (comment out for own orientations)
	auto lineswritten = sim::save_orientation("./orimap", NOF_GRAINS, orientationbuffer.data);
	// check for successful writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d lines written to %s\n",lineswritten, "./orimap");
	*/


	//Load own orimap from txt file (matlab) into orientationbuffer, space as delimiter
	CHECK_FILE_READ(aux::load_ori, "./orimap", orientationbuffer.length, orientationbuffer.data);
	
	/*//Check what is read into orientationbuffer:
	auto lineswritten = sim::save_orientation("./orimap2", NOF_GRAINS, orientationbuffer.data);
	// check for successful writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d lines written to %s\n",lineswritten, "./orimap2");
	*/




	// calculate misorientations
	wt.tic();
	sim::calc_all_misorientation(NOF_GRAINS,
								orientationbuffer,
								misorientation);
	std::printf("time elapsed to calculate misorientations: %fs\n", wt.toc());
	
	// dump binary data to file 
	auto lineswritten = aux::save_array("./misorientation", misorientation.length, misorientation.data);
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d elements written to %s\n", lineswritten, "./misorientation");
	
}
