#ifndef SRC_MESH_STOREMESH3D_HPP 
#define SRC_MESH_STOREMESH3D_HPP 1

#include <src/mesh/mesh3d.hpp>
#include <src/aux/checkfileexists.hpp>
#include <cstdio>
#include <cstddef>

// this macro makes sure that we save the data
// to correct position; matlab uses index 
// starting from 1 but this program uses index zero
#ifndef START_INDEX
#define START_INDEX 1
#endif

namespace mesh { namespace thd {
	
// C-style implementation
template <typename T, typename S, typename R>
int storemesh(char filename[], std::size_t m, std::size_t n, std::size_t k,
			R* nop, 
			std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop, std::ptrdiff_t incLayNop,
			T* val, 
			std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal, std::ptrdiff_t incLayVal,
			S* id, 
			std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId,
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
			for (std::size_t l=0; l<k; ++l) {
				for (std::size_t nl=0; nl<nop[i*incRowNop+j*incColNop+l*incLayNop]; ++nl) {
					// write each line individually
					std::fprintf(fil, "%ld	%ld	%ld	%f	%d\n", 
							i+START_INDEX,j+START_INDEX,l+START_INDEX,
							val[i*incRowVal+j*incColVal+l*incLayVal+nl],
							id[i*incRowId+j*incColId+l*incLayId+nl]);
					// increment numer of lines written
					++lineswritten;
				}
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
int storemesh(char filename[], mesh::thd::Mesh3d<R> &nop, 
			  mesh::thd::Mesh3d<T> &val, mesh::thd::Mesh3d<S> &id,
			  bool overwrite=false
			 ) {
	int lineswritten = storemesh(filename, 
								nop.xsize(), nop.ysize(), nop.zsize(),
								nop.ptr(), nop.xinc(), nop.yinc(), nop.zinc(),
								val.ptr(), val.xinc(), val.yinc(), val.zinc(),
								id.ptr(), id.xinc(), id.yinc(), id.zinc(),
								overwrite
								);
	return lineswritten;
}

// overload function for flat meshes
template <typename T, typename S>
int storemesh(char filename[], std::size_t m, std::size_t n, std::size_t k,
			T* val, 
			std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal, std::ptrdiff_t incLayVal,
			S* id, 
			std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId,
			bool overwrite=false
 			) {
			
	// check if file alread exists
	if (!overwrite && aux::checkfileexists(filename)) {
		std::printf("Error when trying to write file\nFile %s exists\n",filename);
		return -1;
	}
	
	//open file
	FILE *fil = fopen(filename,"w+");
	
	//init return value
	int lineswritten = 0;
	
	// check for success in opening file
	if (fil==NULL) {
		std::printf("Error when trying to open file\nCheck for path\n");
		return -1;
	}
	
	
	// iterate over lines and process input
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<k; ++l) {
				std::fprintf(fil, "%ld	%ld	%ld	%f	%d\n", 
						i+START_INDEX,j+START_INDEX,l+START_INDEX,
						val[i*incRowVal+j*incColVal+l*incLayVal],
						id[i*incRowId+j*incColId+l*incLayId]);
				// increment numer of lines written
				++lineswritten;
			}
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}

//here again the overloaded function
//wrapper function
template <typename T, typename S>
int storemesh(char filename[], mesh::thd::Mesh3d<T> &val, mesh::thd::Mesh3d<S> &id, 
			  bool overwrite=false
			 ) {
	int lineswritten = storemesh(filename, 
								id.xsize(), id.ysize(), id.zsize(),
								val.ptr(), val.xinc(), val.yinc(), val.zinc(),
								id.ptr(), id.xinc(), id.yinc(), id.zinc(),
								overwrite
								);
	return lineswritten;
}

// overload function for flat meshes
// without storing actual values
template <typename S>
int storemesh(char filename[], std::size_t m, std::size_t n, std::size_t k,
			S* id, 
			std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId,
			bool overwrite
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
			for (std::size_t l=0; l<k; ++l) {
				std::fprintf(fil, "%ld	%ld	%ld	%d\n", 
						i+START_INDEX,j+START_INDEX,l+START_INDEX,
						id[i*incRowId+j*incColId+l*incLayId]);
				// increment numer of lines written
				++lineswritten;
			}
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}



//here again the overloaded function
//for storing without actual values
//wrapper function
template <typename S>
int storemesh(char filename[], mesh::thd::Mesh3d<S> &id,
				bool overwrite=false
			 ) {
	int lineswritten = storemesh(filename, 
								id.xsize(), id.ysize(), id.zsize(),
								id.ptr(), id.xinc(), id.yinc(), id.zinc(),
								overwrite
								);
	return lineswritten;
}

// overload function for meshes
// to store positions where the values
// are different from zero
template <typename S, typename T>
int storemesh(char filename[], std::size_t m, std::size_t n, std::size_t k,
			T storagethr,
			S* f, 
			std::ptrdiff_t incRowF, std::ptrdiff_t incColF, std::ptrdiff_t incLayF,
			bool overwrite
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
			for (std::size_t l=0; l<k; ++l) {
				// store positions only
				// if value exceeds some threshold
				if (f[i*incRowF+j*incColF+l*incLayF] > static_cast<S>(storagethr)) {
					std::fprintf(fil, "%ld	%ld	%ld\n", 
							i+START_INDEX,j+START_INDEX,l+START_INDEX
								);
					// increment numer of lines written
					++lineswritten;
				}
			}
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}



//here again the overloaded function
//for storing without actual values
//wrapper function
template <typename S, typename T>
int storemesh(char filename[], 
				T storagethr,
				mesh::thd::Mesh3d<S> &f,
				bool overwrite=false
			 ) {
	return storemesh(filename, 
								f.xsize(), f.ysize(), f.zsize(),
								storagethr,
								f.ptr(), f.xinc(), f.yinc(), f.zinc(),
								overwrite
								);
}
	
} /*namespace thd*/ } /*namespace mesh*/

#endif // SRC_MESH3D_STOREMESH_HPP
