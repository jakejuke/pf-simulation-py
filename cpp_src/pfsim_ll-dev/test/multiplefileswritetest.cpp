#include <cstdio>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <src/mesh/storemesh2d.hpp>
#include <src/aux/walltime.hpp>
#include <omp.h>


// C-style implementation
template <typename T, typename S, typename R>
int storemeshnew(char filename[], std::size_t m, std::size_t n,
			R* nop, std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop,
			T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId) {
			

	int lineswritten = 0;
	//open file
	auto filenamelen = std::strlen(filename);
	std::size_t numthreads = 0;
	#pragma omp parallel shared(numthreads) 
	{
		numthreads = omp_get_num_threads();
	}
	std::vector<FILE*> fil(numthreads);
	for (std::size_t i=0; i<numthreads; ++i) {
		std::sprintf(&filename[filenamelen],"_%ld",i);	
		fil[i] = fopen(filename,"w+");
	}
	
	// iterate over lines and process input
	#pragma omp parallel for schedule(static) reduction(+:lineswritten)
	for (std::size_t i=0; i<m; ++i) {

		auto rank = omp_get_thread_num();
		// iterate over columens and process
		for (std::size_t j=0; j<n; ++j) {

			// iterate over all ops at each lattice site
			for (std::size_t l=0; l<nop[i*incRowNop+j*incColNop]; ++l) {
				// write to buffer
				std::fprintf(fil[rank], 
						"%ld	%ld	%f	%d\n", 
						i+START_INDEX,j+START_INDEX,
						val[i*incRowVal+j*incColVal+l], 
						id[i*incRowId+j*incColId+l]);
				// increment number of lines written
				++lineswritten;

			}

		}
	}

	for (std::size_t i=0; i<numthreads; ++i) {
		fclose(fil[i]);
	}
	return lineswritten;
}

template <typename T, typename S, typename R>
int storemeshnew(char filename[], mesh::twd::Mesh2d<R> &nop, mesh::twd::Mesh2d<T> &val, mesh::twd::Mesh2d<S> &id) {
	int lineswritten = storemeshnew(filename, nop.xsize(), nop.ysize(),
			nop.ptr(), nop.xinc(), nop.yinc(),
			val.ptr(), val.xinc(), val.yinc(),
			id.ptr(), id.xinc(), id.yinc());
	return lineswritten;
}

// C-style implementation
template <typename T, typename S, typename R>
int storemeshcpp(char filename[], std::size_t m, std::size_t n,
		R* nop, std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop,
		T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
		S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId) {

	//open file
	std::ofstream fil;
	fil.open(filename);

	//init return value
	int lineswritten = 0;

	// check for success in opening file


	// iterate over lines and process input

	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {

			for (std::size_t l=0; l<nop[i*incRowNop+j*incColNop]; ++l) {
				// write to buffer
				fil << i+START_INDEX << "\t" 
					<< i+START_INDEX << "\t" 
					<< val[i*incRowVal+j*incColVal+l] << "\t"
					<< id[i*incRowId+j*incColId+l] << "\n";
				++lineswritten;
			}
		}
	}

	fil.close();

	return lineswritten;
}

template <typename T, typename S, typename R>
int storemeshcpp(char filename[], mesh::twd::Mesh2d<R> &nop, mesh::twd::Mesh2d<T> &val, mesh::twd::Mesh2d<S> &id) {
	int lineswritten = storemeshcpp(filename, nop.xsize(), nop.ysize(),
			nop.ptr(), nop.xinc(), nop.yinc(),
			val.ptr(), val.xinc(), val.yinc(),
			id.ptr(), id.xinc(), id.yinc());
	return lineswritten;
}


int main () {
	// 	char a[64] = "aaa";
	// 	char b[10] = "bbb";
	// // 	std::strcat(a, b);
	// 	std::sprintf(a+3,"bbb");
	// 	std::puts(a);


	// configure output directory name
	const char outdirname[32] = "./../workdir/writetest/";

	// define system size
	const std::size_t m = 300;
	const std::size_t n = 300;

	// initialize test case
	mesh::twd::Mesh2d<unsigned short> nopA(m,n);
	mesh::twd::Mesh2d<float> valA(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idA(m,n,0);

	// set up timer
	aux::WallTime<float> wt;

	// read from file to init mesh
	char infilename[64] = "./../workdir/testcase/partition_step100";
	auto linesread = mesh::twd::init_from_file(infilename, nopA, valA, idA);
	std::printf("lines read for mesh init: %d\n", linesread);

#pragma omp parallel
	{
		std::printf("hi from the threads\n");
	}

	// Write the original dist
	char outfilename[128] = "./../workdir/writetest/resold";
	// call function to store
	wt.tic();
	int lineswritten;
	lineswritten = mesh::twd::storemesh(outfilename, nopA, valA, idA);

	for (std::size_t i=0; i<10; ++i) {
		std::sprintf(outfilename, "%sresnew", outdirname);
		lineswritten = storemeshnew(outfilename, nopA, valA, idA);
	}	
	std::printf("%d lines written to %s in %fs\n",lineswritten, outfilename, wt.toc());

	mesh::twd::Mesh2d<unsigned short> nopB(m,n);
	mesh::twd::Mesh2d<float> valB(m,n,0);
	mesh::twd::Mesh2d<unsigned short> idB(m,n,0);

	linesread = mesh::twd::init_from_file(outfilename, nopB, valB, idB);

	char checkout[64] = "./../workdir/writetest/rescheck";
	lineswritten = mesh::twd::storemesh(checkout, nopB, valB, idB);

}
