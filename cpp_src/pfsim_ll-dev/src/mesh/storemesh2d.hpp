#ifndef SRC_MESH_STOREMESH2D_HPP 
#define SRC_MESH_STOREMESH2D_HPP 1

#include <src/mesh/mesh2d.hpp>
#include <src/aux/checkfileexists.hpp>
#include <cstdio>
#include <cstddef>

// this macro makes sure that we save the data
// to correct position; matlab uses index 
// starting from 1 but this program uses index zero
#ifndef START_INDEX
#define START_INDEX 1
#endif

namespace mesh { namespace twd {
	
// C-style implementation
template <typename T, typename S, typename R>
int storemesh(char filename[], std::size_t m, std::size_t n,
			R* nop, std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop,
			T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId,
			bool overwrite=false
 			) {
			
	// check if file alread exists
	if (!overwrite && aux::checkfileexists(filename)) {
		std::printf("Error when trying to write file\nFile %s exists\n",filename);
		return -1;
	}
	
	//open file
	FILE *fil = fopen(filename,"w+");
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("(Zeile 38) Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	//init return value
	int lineswritten = 0;

	
	// iterate over lines and process input
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<nop[i*incRowNop+j*incColNop]; ++l) {
				// write each line individually
				std::fprintf(fil, "%ld	%ld	%f	%d\n", i+START_INDEX,j+START_INDEX,
						val[i*incRowVal+j*incColVal+l],id[i*incRowId+j*incColId+l]);
				// increment numer of lines written
				++lineswritten;
			}
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}

//wrapper function
/* TODO: write it in a style that lets
 * instanciate for Mesh2d and Mesh2dView
 */
template <typename T, typename S, typename R>
int storemesh(char filename[], mesh::twd::Mesh2d<R> &nop, mesh::twd::Mesh2d<T> &val, mesh::twd::Mesh2d<S> &id, bool overwrite=false) {
	return storemesh(filename, nop.xsize(), nop.ysize(),
									nop.ptr(), nop.xinc(), nop.yinc(),
									val.ptr(), val.xinc(), val.yinc(),
									id.ptr(), id.xinc(), id.yinc(),
									overwrite
					);
}

// overload function for flat meshes
template <typename T, typename S>
int storemesh(char filename[], std::size_t m, std::size_t n,
			T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId,
			bool overwrite=false
 			) {
			
	// check if file alread exists
	if (!overwrite && aux::checkfileexists(filename)) {
		std::printf("Error when trying to write file\nFile %s exists\n",filename);
		return -1;
	}
	
	//open file
	FILE *fil = fopen(filename,"w+");
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	//init return value
	int lineswritten = 0;
	
	// iterate over lines and process input
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			// write each line individually
			std::fprintf(fil, "%ld	%ld	%f	%d\n", i+START_INDEX,j+START_INDEX,
					val[i*incRowVal+j*incColVal],id[i*incRowId+j*incColId]);
			// increment numer of lines written
				++lineswritten;
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}

//here again the overloaded function
//wrapper function
template <typename T, typename S>
int storemesh(char filename[], mesh::twd::Mesh2d<T> &val, mesh::twd::Mesh2d<S> &id,
				bool overwrite=false
				) {
	return storemesh(filename, id.xsize(), id.ysize(),
									val.ptr(), val.xinc(), val.yinc(),
									id.ptr(), id.xinc(), id.yinc(),
									overwrite
					);
}

// overload function for flat meshes
// and not storing the values but id only
template <typename S>
int storemesh(const char filename[], std::size_t m, std::size_t n,
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId,
			bool overwrite=false
 			) {
			
	// check if file alread exists
	if (!overwrite && aux::checkfileexists(filename)) {
		std::printf("Error when trying to write file\nFile %s exists\n",filename);
		return -1;
	}
	
	//open file
	FILE *fil = fopen(filename,"w+");
	
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	//init return value
	int lineswritten = 0;
	
	// iterate over lines and process input
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			// write each line individually
			std::fprintf(fil, "%ld	%ld	%d\n", i+START_INDEX,j+START_INDEX,
					id[i*incRowId+j*incColId]);
			// increment numer of lines written
				++lineswritten;
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}


//here again the overloaded function
//and not storing values
//wrapper function
template <typename S>
int storemesh(const char filename[], mesh::twd::Mesh2d<S> &id,
				bool overwrite=false
			 ) {
	return storemesh(filename, id.xsize(), id.ysize(),
					id.ptr(), id.xinc(), id.yinc(),
					overwrite
					);
}

// overload function for meshes
// to store positions where the values are 
// different from zero
template <typename S, typename T>
int storemesh(char filename[], 
			std::size_t m, std::size_t n,
			T storagethr,
			S* f, std::ptrdiff_t incRowF, std::ptrdiff_t incColF,
			bool overwrite=false
 			) {
			
	// check if file alread exists
	if (!overwrite && aux::checkfileexists(filename)) {
		std::printf("Error when trying to write file\nFile %s exists\n",filename);
		return -1;
	}
	
	//open file
	FILE *fil = fopen(filename,"w+");
	
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	//init return value
	int lineswritten = 0;
	
	// iterate over lines and process input
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			// store positions only
			// if value exceeds some threshold
			if (f[i*incRowF+j*incColF] > static_cast<S>(storagethr)) {
				// write each line individually
				std::fprintf(fil, "%ld	%ld\n", i+START_INDEX,j+START_INDEX);
				// increment numer of lines written
				++lineswritten;
			} 
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}


//here again the overloaded function
//and not storing values, but positons only
//wrapper function
template <typename S, typename T>
int storemesh(char filename[], T storagethr,
				mesh::twd::Mesh2d<S> &f,
				bool overwrite=false
			 ) {
	return storemesh(filename, f.xsize(), f.ysize(),
					storagethr,
					f.ptr(), f.xinc(), f.yinc(),
					overwrite
					);
}
	
} /*namespace twd*/ } /*namespace mesh*/

#endif // SRC_MESH2D_STOREMESH_HPP
