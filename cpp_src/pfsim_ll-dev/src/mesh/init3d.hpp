#ifndef SRC_MESH_INIT3D_HPP 
#define SRC_MESH_INIT3D_HPP 1

#include <src/mesh/mesh3d.hpp>
#include <src/mesh/compat3d.hpp>
#include <cstdio>
#include <cassert>

// this macro makes sure that we load the data
// to correct position; matlab uses index 
// starting from 1 but this program uses index zero
#ifndef START_INDEX
#define START_INDEX 1
#endif

namespace mesh { namespace thd {
	
//C-style implementation
template <typename T, typename S, typename R>
int init_from_file(char filename[], 
	std::size_t m, std::size_t n, std::size_t k, std::size_t nop,
	R *nopdest,
	std::ptrdiff_t incRowNopdest, std::ptrdiff_t incColNopdest, std::ptrdiff_t incLayNopdest,
	T *dest, 
	std::ptrdiff_t incRowDest, std::ptrdiff_t incColDest, std::ptrdiff_t incLayDest,
	S *iddest, 
	std::ptrdiff_t incRowIddest, std::ptrdiff_t incColIddest, std::ptrdiff_t incLayIddest) {
	
	FILE *fil = fopen(filename,"r");
	std::size_t linesread = 0;
	
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	// local vars to write input to
	int i,j,l,id;
	float val;
	int current_nop;
	
	// iterate over lines and process input
	// TODO: optimization by loading data in chunks
	// and processing it afterwards
	while(!feof(fil)) {
		
		//read line-by-line and check for errors
		if(std::fscanf(fil, "%d %d %d %f %d\n", &i, &j, &l, &val, &id) != 5) {
			std::printf("error when reading lines\nCheck for file content\n");
			return -1;
		}
		
		//adjust indexing
		i-=START_INDEX; j-=START_INDEX; l-=START_INDEX;
		
		// fetch current nop at input position;
		// use it to store the loaded value,
		// at the same time increment nop
		current_nop = nopdest[i*incRowNopdest+j*incColNopdest+l*incLayNopdest]++;
		
		//store value 
		dest[i*incRowDest+j*incColDest+l*incLayDest+current_nop] = val;
		
		//store id
		iddest[i*incRowIddest+j*incColIddest+l*incLayIddest+current_nop] = id;
		
		++linesread;
	}
	
	fclose(fil);
	
	return linesread;
}
					
//now use mesh as input and then call C-style function
template <typename T, typename S, typename R>
int init_from_file(char filename[], mesh::thd::Mesh3d<R> &nopdest, mesh::thd::Mesh3d<T> &dest, mesh::thd::Mesh3d<S> &iddest) {
	
	// assert the compatibility of input 
	// mesh objects
	assert(mesh::thd::check_ident(dest, iddest));
	assert(mesh::thd::check_compat(dest, nopdest));
	
	//call C-style function
	int linesread = init_from_file(filename, 
					nopdest.xsize(), nopdest.ysize(), nopdest.zsize(), 
					dest.opbufsize(),
					nopdest.ptr(),
					nopdest.xinc(), nopdest.yinc(), nopdest.zinc(),
					dest.ptr(),
					dest.xinc(), dest.yinc(), dest.zinc(),
					iddest.ptr(),
					iddest.xinc(), iddest.yinc(), iddest.zinc());

	return linesread;
}

// overload function to load
// files that contain information on
// particlefields, i.e. we have to
// specify the values also
template <typename R>
int init_from_file(char filename[],
	std::size_t m, std::size_t n, std::size_t k,
	R *f,
	std::ptrdiff_t incRowF, std::ptrdiff_t incColF, std::ptrdiff_t incLayF,
	R epsilon) {
	
	FILE *fil = fopen(filename,"r");
	std::size_t linesread = 0;
	
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	// local vars to write input to
	int i,j,l;
	
	//iterate over lines and process inout
	while(!feof(fil)) {
		
		//read line-by-line and check errors
		if(std::fscanf(fil, "%d %d %d\n", &i, &j, &l) != 3) {
			std::printf("error when reading lines\nCheck for file content\n");
			return -1;
		}
		
		// adjusting indexing
		i-=START_INDEX; j-=START_INDEX; l-=START_INDEX;
		
		//store id
		f[i*incRowF+j*incColF+l*incLayF] = epsilon;
		
		++linesread;
	}
	
	fclose(fil);
	
	return linesread;
}

//now use mesh as input and then call C-style function
template <typename S>
int init_from_file(char filename[],
				   mesh::thd::Mesh3d<S> &iddest,
				   S epsilon
				) {
	// call C-style function
	int linesread = init_from_file(filename,
					iddest.xsize(), iddest.ysize(), iddest.zsize(),
					iddest.ptr(),
					iddest.xinc(), iddest.yinc(), iddest.zinc(),
					epsilon);
	return linesread;
}

//TODO: function for random initialization

	
} /*namespace thd*/ } /*namespace mesh*/

#endif // SRC_MESH3D_INIT_HPP
