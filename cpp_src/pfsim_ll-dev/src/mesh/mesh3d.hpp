#ifndef SRC_MESH_MESH3D_HPP
#define SRC_MESH_MESH3D_HPP 1

// cache alignment macro
// should be equal to cacheline
// size or half of it
#ifndef CACHE_LINE_SIZE_DEFAULT
#define CACHE_LINE_SIZE_DEFAULT 64
#endif

#include <cstddef>
#include <cassert>
#include <src/aux/alignmem.hpp>

namespace mesh { namespace thd {
	
template <typename T>
class Mesh3d {
	/* general idea:
	 * save the number of nonzero ops on the
	 * points with i and j, starting from there
	 * have 64 fields of space for the values of
	 * the order parameters,
	 * fields can be accessed via objectname(x,y,op);
	 * number of nonzero fields can be accessed by 
	 * objectname(x,y);
	*/
	
	//TODO: implement mechanism that prevents incorrect
	// use for access of a "flat"/"unflat" mesh, rspctly

public:	
	// standard constructor;
	// will basically create a matrix if
	// buffer depth not specified
	// used to allocate memory
	Mesh3d(std::size_t m, std::size_t n, std::size_t k) : nopmax(1), 
		numRows(m), numCols(n), numLays(k), 
		incRow(n*k), incCol(k), incLay(1), 
		flat(true), 
		mem(new T[aux::compute_aligned_size<T>(numRows*numCols*numLays*nopmax, CACHE_LINE_SIZE_DEFAULT)]()),
		data(aux::align_ptr(mem, CACHE_LINE_SIZE_DEFAULT))
	{
	}
	
	// specified buffer size, corresponds 
	// to adding a third dimension;
	Mesh3d(std::size_t m, std::size_t n, std::size_t k, 
		   std::size_t localdim) : 
		nopmax(localdim), 
		numRows(m), numCols(n), numLays(k), 
		incRow(n*k*nopmax), incCol(k*nopmax), incLay(nopmax), 
		flat(localdim==1? true : false), 
		mem(new T[aux::compute_aligned_size<T>(numRows*numCols*numLays*nopmax, CACHE_LINE_SIZE_DEFAULT)]()),
		data(aux::align_ptr(mem, CACHE_LINE_SIZE_DEFAULT))
	{
	}
	
	// standard destructor
	// to release memory
	~Mesh3d() 
	{
		delete[] mem;
	}
	
	void memset(T value) {
		for (std::size_t i=0; i<numRows*numCols*numLays*nopmax; ++i) {
			data[i] = value;
		}
	}
	
	// read-only access methods for 
	// order parameter values
	const T& operator() (std::size_t i, std::size_t j, std::size_t k, std::size_t op) const 
	{
		//assert(!flat);
		assert(i>=0 && i<numRows && j>=0 && j<numCols && k>=0 && k<numLays);
		assert(op>=0 && op<nopmax);
		return data[i*incRow+j*incCol+k*incLay+op];
	}
	
	// write access methods
	T& operator() (std::size_t i, std::size_t j, std::size_t k, std::size_t op) 
	{
		//assert(!flat);
		assert(i>=0 && i<numRows && j>=0 && j<numCols && k>=0 && k<numLays);
		assert(op>=0 && op<nopmax);
		return data[i*incRow+j*incCol+k*incLay+op];
	}
	
	// read-only access methods for 
	// flat mesh
	const T& operator() (std::size_t i, std::size_t j, std::size_t k) const 
	{
		//assert(flat);
		assert(i>=0 && i<numRows && j>=0 && j<numCols && k>=0 && k<numLays);
		return data[i*incRow+j*incCol+k*incLay];
	}
	
	// write access methods
	T &operator() (std::size_t i, std::size_t j, std::size_t k) 
	{
		//assert(flat);
		assert(i>=0 && i<numRows && j>=0 && j<numCols && k>=0 && k<numLays);
		return data[i*incRow+j*incCol+k*incLay];
	}

	
	// method to access pointer to data
	// might come in handy for some algorithms
	// but be sure to know what you do
	T* ptr() const {
		return data;
	}
	
	T* memptr() const {
		return mem;
	}
	
	// method to reset pointer, e.g. for swap
	void set_ptr(T *new_data) {
		data = new_data;
	}
	
	void set_memptr(T* new_mem, T* new_data) {
		mem = new_mem;
		data = new_data;
	}
	
	// methods to infer on size
	std::size_t xsize() const {
		return numRows;
	}
	
	std::size_t ysize() const {
		return numCols;
	}
	
	std::size_t zsize() const {
		return numLays;
	}
	
	std::size_t opbufsize() const {
		return nopmax;
	}
	
	std::size_t xinc() const {
		return incRow;
	}
	
	std::size_t yinc() const {
		return incCol;
	}
	
	std::size_t zinc() const {
		return incLay;
	}
	
	// delete copy constructor and assignment operator
	// so that no hidden errors can occur
	Mesh3d& operator=(const Mesh3d&) = delete;
    Mesh3d(const Mesh3d&) = delete;
	
private:
	// increments used for access
	// and mesh size
	std::size_t nopmax;
	std::size_t numRows, numCols, numLays;
	std::ptrdiff_t incRow, incCol, incLay;
	
	// 2 or 3 for 2D mesh or 3D mesh, might 
	// come in handy for MPI later
	//std::size_t dim;
	
	//attribute to check if it is "flat"
	bool flat;
	
	// pointer to data
	T *mem;
	T *data;
};

} /*namespace thd*/ } /*namespace mesh*/ 


#endif // SRC_MESH_MESH3D_HPP
