#include <src/mesh3d.hpp>
#include <src/sim3d.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/walltime.hpp>
#include <cstdio>
#include <cstddef>

// number of iterations
#ifndef NUM_ITER
#define NUM_ITER 10
#endif

// num of iterations after which output is generated
#ifndef STEP_SIZE
#define STEP_SIZE 1
#endif

// overwrite flag
#ifndef OVERWRITEFLAG
#define OVERWRITEFLAG true
#endif

template <typename R, typename S, typename T>
void fill_ball(mesh::thd::Mesh3d<R>& nop, mesh::thd::Mesh3d<T>& val, mesh::thd::Mesh3d<S>& id) {
	
	// generate coordinates for center of ball
	// take into account, that position coordinates
	// each range [0,nop.*size()-1]
	double centercoords[3] = {(double) (nop.xsize()-1)/2,
								(double) (nop.ysize()-1)/2,
								(double) (nop.zsize()-1)/2};

	// get radius: 40% of half diameter of smallest 
	// side
	double radius2 = 0.4*centercoords[0];
	for (std::size_t i=0; i<3; ++i) {
		if (0.4*centercoords[i] > radius2) {
			radius2 = centercoords[i]*0.4;
		}
	}
	//subs with the value squared
	radius2 = radius2*radius2;
	
	// loop over all dimensions and fill meshes
	for (std::size_t i=0; i<nop.xsize(); ++i) {
		for (std::size_t j=0; j<nop.ysize(); ++j) {
			for (std::size_t l=0; l<nop.ysize(); ++l) {
				
				// immediately fill nop and val;
				// sharp boundary style
				nop(i,j,l) = 1;
				val(i,j,l,0) = 1.0;
				
				// determine currents position distance
				// to ball center
				double dist_to_center2 = (i-centercoords[0])*(i-centercoords[0]) 
								+ (j-centercoords[1])*(j-centercoords[1])
								+ (l-centercoords[2])*(l-centercoords[2]);
				
				// if the distance is smaller, then 
				// set id=1, else id=2
				if (dist_to_center2 <= radius2) {
					id(i,j,l,0) = 1;
				} else {
					id(i,j,l,0) = 2;
				}
			}
		}
	}
}

template <typename R, typename S, typename T>
void fill_uni(mesh::thd::Mesh3d<R>& nop, mesh::thd::Mesh3d<T>& val, mesh::thd::Mesh3d<S>& id) {
	
	// fill nop
	nop.memset(1);
	
	// loop over all dimensions and fill meshes
	for (std::size_t i=0; i<nop.xsize(); ++i) {
		for (std::size_t j=0; j<nop.ysize(); ++j) {
			for (std::size_t l=0; l<nop.ysize(); ++l) {
				
				// fill nop and val uniformly
				id(i,j,l) = 1;
				val(i,j,l,0) = 1.0;
			}
		}
	}
}


int main() {
	
	// configure output directory name
	const char outdirname[32] = "./../workdir/3dperturbtest/";

	// define system size
	const std::size_t m = 5;
	const std::size_t n = 5;
	const std::size_t k = 5;
	
	// initialize test case
	mesh::thd::Mesh3d<unsigned short> nopA(m,n,k);
	mesh::thd::Mesh3d<float> valA(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> idA(m,n,k,0);
	
	// create the meshes that we propagate to
	mesh::thd::Mesh3d<unsigned short> nopB(m,n,k);
	mesh::thd::Mesh3d<float> valB(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> idB(m,n,k,0);
	
	// create meshes to store maximum of each op
	mesh::thd::Mesh3d<float> valMax(m,n,k);
	mesh::thd::Mesh3d<unsigned short> idMax(m,n,k);
	
	// init mesh
	fill_uni(nopA, valA, idA);
	// include small perturbation
	idA(2,2,2) = 2;
	
	// Write the original dist
	char outfilename[128];
	std::sprintf(outfilename, "%spartition_step000", outdirname);
	// call function to store
	auto lineswritten = mesh::thd::storemesh(outfilename, nopA, valA, idA, OVERWRITEFLAG);
	// and check for success of writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	
	std::printf("%d lines written to %s\n",lineswritten, outfilename);
	std::sprintf(outfilename, "%slm_partition_step000", outdirname);
	// call function to store
	mesh::thd::maxop(nopA, valA, idA, idMax);
	lineswritten = mesh::thd::storemesh(outfilename, idMax, OVERWRITEFLAG);
	// and check for success of writing
	if (lineswritten==-1 || lineswritten==0) {
		std::printf("data file output error: %d, aborting\n", lineswritten);
		return (2);
	}
	std::printf("%d lines written to %s\n",lineswritten, outfilename);

	
	// set up timer
	aux::WallTime<float> wt;
	wt.tic();

	for (std::size_t l=1; l<=NUM_ITER; ++l) {
		// perform propagation of the complete mesh
		auto status = sim::thd::stepiso(nopA, valA, idA, 
				nopB, valB, idB);
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
			lineswritten = mesh::thd::storemesh(outfilename, nopB, valB, idB, OVERWRITEFLAG);
			// and check for success of writing
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error: %d, aborting\n", lineswritten);
				return (2);
			}
			std::printf("%d lines written to %s in %fs\n",lineswritten, outfilename, wt.toc());
			
			// output of local max
			mesh::thd::maxop(nopB, valB, idB, idMax);
			
			// generate filename
			std::sprintf(outfilename, "%slm_partition_step%03ld", outdirname, l);
			
			//call function to store results
			wt.tic();
			lineswritten = mesh::thd::storemesh(outfilename, idMax, OVERWRITEFLAG);
			// and check for success of writing
			if (lineswritten==-1 || lineswritten==0) {
				std::printf("data file output error: %d, aborting\n", lineswritten);
				return (2);
			}
			std::printf("%d lines written to %s in %fs\n",lineswritten, outfilename, wt.toc());
			
			// reset start timer
			wt.tic();
		}
		

		// swap the pointers to propagate in the
		// other direction now
		mesh::swap(nopA, nopB);
		mesh::swap(valA, valB);
		mesh::swap(idA, idB);
	}

	std::printf("exit with success\n");
}
