#ifndef SRC_MESH_PSTOREMESH2D_HPP 
#define SRC_MESH_PSTOREMESH2D_HPP 1

#include <src/mesh/mesh2d.hpp>
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
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId) {
			
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

            // iterate over all ops at each lattice site
			for (std::size_t l=0; l<nop[i*incRowNop+j*incColNop]; ++l) {
				// write to buffer
				charcounter += std::sprintf(&strbuf[charcounter], 
					"%ld	%ld	%f	%d\n", 
					i+START_INDEX,j+START_INDEX,
					val[i*incRowVal+j*incColVal+l], 
					id[i*incRowId+j*incColId+l]);
				// increment number of lines written
				++lineswritten;
				
				//std::strcat(strbuf,smallbuf);
				
				if (charcounter >  bufsize-19) {
					#pragma omp critical 
					{
					std::fputs(strbuf, fil);
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
int storemesh(char filename[], std::size_t m, std::size_t n,
			T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId) {
			
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

			// write to buffer
			charcounter += std::sprintf(&strbuf[charcounter], 
				"%ld	%ld	%f	%d\n", 
				i+START_INDEX,j+START_INDEX,
				val[i*incRowVal+j*incColVal], 
				id[i*incRowId+j*incColId]);
			// increment number of lines written
			++lineswritten;
			
			//std::strcat(strbuf,smallbuf);
			
			if (charcounter >  bufsize-19) {
				#pragma omp critical 
				{
				std::fputs(strbuf, fil);
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
// and not storing the values but id only
template <typename S>
int storemesh(char filename[], std::size_t m, std::size_t n,
			S* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId) {
			
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

			// write to buffer
			charcounter += std::sprintf(&strbuf[charcounter], 
				"%ld	%ld	%d\n", 
				i+START_INDEX,j+START_INDEX,
				id[i*incRowId+j*incColId]);
			// increment number of lines written
			++lineswritten;
			
			//std::strcat(strbuf,smallbuf);
			
			if (charcounter >  bufsize-19) {
				#pragma omp critical 
				{
				std::fputs(strbuf, fil);
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
int storemesh(char filename[], mesh::twd::Mesh2d<R> &nop, mesh::twd::Mesh2d<T> &val, mesh::twd::Mesh2d<S> &id) {
	return storemesh(filename, nop.xsize(), nop.ysize(),
									nop.ptr(), nop.xinc(), nop.yinc(),
									val.ptr(), val.xinc(), val.yinc(),
									id.ptr(), id.xinc(), id.yinc());
}

//here again the overloaded function
//wrapper function
template <typename T, typename S>
int storemesh(char filename[], mesh::twd::Mesh2d<T> &val, mesh::twd::Mesh2d<S> &id) {
	return storemesh(filename, id.xsize(), id.ysize(),
									val.ptr(), val.xinc(), val.yinc(),
									id.ptr(), id.xinc(), id.yinc());
}

//here again the overloaded function
//and not storing values
//wrapper function
template <typename S>
int storemesh(char filename[], mesh::twd::Mesh2d<S> &id) {
	return storemesh(filename, id.xsize(), id.ysize(),
									id.ptr(), id.xinc(), id.yinc());
}
	
} /*namespace twd*/ } /*namespace mesh*/

#endif // SRC_MESH_STOREMESHPAR2D_HPP
