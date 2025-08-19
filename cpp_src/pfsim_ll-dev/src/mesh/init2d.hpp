#ifndef SRC_MESH_INIT2D_HPP 
#define SRC_MESH_INIT2D_HPP 1

#include <src/mesh/mesh2d.hpp>
#include <src/mesh/compat2d.hpp>
#include <cstdio>
#include <cassert>

// this macro makes sure that we load the data
// to correct position; matlab uses index 
// starting from 1 but this program uses index zero
#ifndef START_INDEX
#define START_INDEX 1
#endif

namespace mesh { namespace twd {
	
//C-style implementation
template <typename T, typename S, typename R>
int init_from_file(char filename[], 
	std::size_t m, std::size_t n, std::size_t nop,
	R *nopdest,
	std::ptrdiff_t incRowNopdest, std::ptrdiff_t incColNopdest,
	T *dest, 
	std::ptrdiff_t incRowDest, std::ptrdiff_t incColDest,
	S *iddest, 
	std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest) {
	
	FILE *fil = fopen(filename,"r");
	std::size_t linesread = 0;
	
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	// local vars to write input to
	int i,j,id;
	float val;
	int current_nop;
	
	// iterate over lines and process input
	while(!feof(fil)) {
		
		//read line-by-line and check for errors
		if(std::fscanf(fil, "%d %d %f %d\n", &i, &j, &val, &id) != 4) {
			std::printf("error when reading lines\nCheck for file content\n");
			return -1;
		}
		
		//adjust indexing
		i-=START_INDEX; j-=START_INDEX;
		
		// fetch current nop at input position;
		// use it to store the loaded value,
		// at the same time increment nop
		current_nop = nopdest[i*incRowNopdest+j*incColNopdest]++;
		
		//store value 
		dest[i*incRowDest+j*incColDest+current_nop] = val;
		
		//store id
		iddest[i*incRowIddest+j*incColIddest+current_nop] = id;
		
		++linesread;
	}
	
	fclose(fil);
	
	return linesread;
}
					
//now use mesh as input and then call C-style function
template <typename T, typename S, typename R>
int init_from_file(char filename[], mesh::twd::Mesh2d<R> &nopdest, mesh::twd::Mesh2d<T> &dest, mesh::twd::Mesh2d<S> &iddest) {
	
	// assert the compatibility of input 
	// mesh objects
	assert(mesh::twd::check_ident(dest, iddest));
	assert(mesh::twd::check_compat(dest, nopdest));
	
	//call C-style function
	int linesread = init_from_file(filename, 
					nopdest.xsize(), nopdest.ysize(), nopdest.opbufsize(),
					nopdest.ptr(),
					nopdest.xinc(), nopdest.yinc(),
					dest.ptr(),
					dest.xinc(), dest.yinc(),
					iddest.ptr(),
					iddest.xinc(), iddest.yinc());

	return linesread;
}

//overload function to load
// files that contain information on 
// local max, i.e. we only need a single
// flat mesh to store the ids
template <typename R>
int init_from_file(char filename[], 
	std::size_t m, std::size_t n,
	R *iddest,
	std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest) {
	
	FILE *fil = fopen(filename,"r");
	std::size_t linesread = 0;
	
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	// local vars to write input to
	int i,j,id;
	
	// iterate over lines and process input
	while(!feof(fil)) {
		
		//read line-by-line and check for errors
		if(std::fscanf(fil, "%d %d %d\n", &i, &j, &id) != 3) {
			std::printf("error when reading lines\nCheck for file content\n");
			return -1;
		}
		
		//adjust indexing
		i-=START_INDEX; j-=START_INDEX;
		
		//store id
		iddest[i*incRowIddest+j*incColIddest] = id;
		
		++linesread;
	}
	
	fclose(fil);
	
	return linesread;
}
					
//now use mesh as input and then call C-style function
template <typename S>
int init_from_file(char filename[], mesh::twd::Mesh2d<S> &iddest) {
	
	//call C-style function
	int linesread = init_from_file(filename, 
					iddest.xsize(), iddest.ysize(),
					iddest.ptr(),
					iddest.xinc(), iddest.yinc());

	return linesread;
}

//overload function to load
// files that contain information on
// particlefields, i.e. we have to
// specify the values also
template <typename R>
int init_from_file(char filename[], 
	std::size_t m, std::size_t n,
	R *f,
	std::ptrdiff_t incRowF, std::ptrdiff_t incColF,
	R epsilon) {
	
	FILE *fil = fopen(filename,"r");
	std::size_t linesread = 0;
	
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	// local vars to write input to
	int i,j;
	
	// iterate over lines and process input
	while(!feof(fil)) {
		
		//read line-by-line and check for errors
		if(std::fscanf(fil, "%d %d\n", &i, &j) != 2) {
			std::printf("error when reading lines\nCheck for file content\n");
			return -1;
		}
		
		//adjust indexing
		i-=START_INDEX; j-=START_INDEX;
		
		//store id
		f[i*incRowF+j*incColF] = epsilon;
		
		++linesread;
	}
	
	fclose(fil);
	
	return linesread;
}
					
//now use mesh as input and then call C-style function
template <typename S>
int init_from_file(char filename[], 
				   mesh::twd::Mesh2d<S> &iddest,
				   S epsilon
  				) {
	
	//call C-style function
	int linesread = init_from_file(filename, 
					iddest.xsize(), iddest.ysize(),
					iddest.ptr(),
					iddest.xinc(), iddest.yinc(),
					epsilon);

	return linesread;
}

//TODO: function for random initialization

	
} /*namespace twd*/ } /*namespace mesh*/

#endif // SRC_MESH2D_INIT_HPP
