#ifndef SRC_MESH_PSTOREMESH3D_HPP 
#define SRC_MESH_PSTOREMESH3D_HPP 1

#include <src/mesh/mesh3d.hpp>
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
			std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId
			 ) {
			
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
	const int bufsize = 512;
	#pragma omp parallel for schedule (static) reduction(+:lineswritten)
	for (std::size_t i=0; i<m; ++i) {
		char strbuf[bufsize];
		strbuf[0] = '\0';
        int charcounter = 0;
		
        // iterate over columens and process
		for (std::size_t j=0; j<n; ++j) {
			
			for (std::size_t l=0; l<k; ++l) {

				// iterate over all ops at each lattice site
				for (std::size_t nl=0; nl<nop[i*incRowNop+j*incColNop+l*incLayNop]; ++nl) {
					// write to buffer
					charcounter += std::sprintf(&strbuf[charcounter], 
						"%ld	%ld	 %ld	%f	%d\n", 
						i+START_INDEX,j+START_INDEX,l+START_INDEX,
						val[i*incRowVal+j*incColVal+l*incLayVal+nl], 
						id[i*incRowId+j*incColId+l*incLayId+nl]);
					// increment number of lines written
					++lineswritten;
					
					//std::strcat(strbuf,smallbuf);
					
					if (charcounter >  bufsize-19) {
						#pragma omp critical 
						{
							std::fputs(strbuf, fil);
						}
						strbuf[0] = '\0';
						charcounter = 0;
					}
				}
			}
			
		}
		
		if (charcounter > 0) {
			#pragma omp critical
			{
				std::fputs(strbuf, fil);
			}
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}

// overload function for flat meshes
template <typename T, typename S>
int storemesh(char filename[], std::size_t m, std::size_t n, std::size_t k,
			T* val, 
			std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal, std::ptrdiff_t incLayVal,
			S* id, 
			std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId) {
			
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
	const int bufsize = 512;
	#pragma omp parallel for schedule (static) reduction(+:lineswritten)
	for (std::size_t i=0; i<m; ++i) {
		char strbuf[bufsize];
		strbuf[0] = '\0';
        int charcounter = 0;
		
        // iterate over columens and process
		for (std::size_t j=0; j<n; ++j) {
			
			for (std::size_t l=0; l<k; ++l) {

				
				// write to buffer
				charcounter += std::sprintf(&strbuf[charcounter], 
					"%ld	%ld	 %ld	%f	%d\n", 
					i+START_INDEX,j+START_INDEX,l+START_INDEX,
					val[i*incRowVal+j*incColVal+l*incLayVal], 
					id[i*incRowId+j*incColId+l*incLayId]);
				// increment number of lines written
				++lineswritten;
				
				//std::strcat(strbuf,smallbuf);
				
				if (charcounter >  bufsize-19) {
					#pragma omp critical 
					{
						std::fputs(strbuf, fil);
					}
					strbuf[0] = '\0';
					charcounter = 0;
				}
			
			}
			
		}
		
		if (charcounter > 0) {
			#pragma omp critical
			{
				std::fputs(strbuf, fil);
			}
		}
	}
	
	fclose(fil);
	
	return lineswritten;
}

// overload function for flat meshes
// without storing actual values
template <typename S>
int storemesh(char filename[], std::size_t m, std::size_t n, std::size_t k,
			S* id, 
			std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId) {
			
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
	const int bufsize = 512;
	#pragma omp parallel for schedule (static) reduction(+:lineswritten)
	for (std::size_t i=0; i<m; ++i) {
		char strbuf[bufsize];
		strbuf[0] = '\0';
        int charcounter = 0;
		
        // iterate over columens and process
		for (std::size_t j=0; j<n; ++j) {
			
			for (std::size_t l=0; l<k; ++l) {

				
				// write to buffer
				charcounter += std::sprintf(&strbuf[charcounter], 
					"%ld	%ld	 %ld	%d\n", 
					i+START_INDEX,j+START_INDEX,l+START_INDEX,
					id[i*incRowId+j*incColId+l*incLayId]);
				// increment number of lines written
				++lineswritten;
				
				//std::strcat(strbuf,smallbuf);
				
				if (charcounter >  bufsize-19) {
					#pragma omp critical 
					{
						std::fputs(strbuf, fil);
					}
					strbuf[0] = '\0';
					charcounter = 0;
				}
			
			}
			
		}
		
		if (charcounter > 0) {
			#pragma omp critical
			{
				std::fputs(strbuf, fil);
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
int storemesh(char filename[], mesh::thd::Mesh3d<R> &nop, mesh::thd::Mesh3d<T> &val, mesh::thd::Mesh3d<S> &id) {
	int lineswritten = storemesh(filename, 
								nop.xsize(), nop.ysize(), nop.zsize(),
								nop.ptr(), nop.xinc(), nop.yinc(), nop.zinc(),
								val.ptr(), val.xinc(), val.yinc(), val.zinc(),
								id.ptr(), id.xinc(), id.yinc(), id.zinc());
	return lineswritten;
}

//here again the overloaded function
//wrapper function
template <typename T, typename S>
int storemesh(char filename[], mesh::thd::Mesh3d<T> &val, mesh::thd::Mesh3d<S> &id) {
	int lineswritten = storemesh(filename, 
								id.xsize(), id.ysize(), id.zsize(),
								val.ptr(), val.xinc(), val.yinc(), val.zinc(),
								id.ptr(), id.xinc(), id.yinc(), id.zinc());
	return lineswritten;
}

//here again the overloaded function
//for storing without actual values
//wrapper function
template <typename S>
int storemesh(char filename[], mesh::thd::Mesh3d<S> &id) {
	int lineswritten = storemesh(filename, 
								id.xsize(), id.ysize(), id.zsize(),
								id.ptr(), id.xinc(), id.yinc(), id.zinc());
	return lineswritten;
}
	
} /*namespace thd*/ } /*namespace mesh*/

#endif // SRC_MESH_STOREMESHPAR3D_HPP
