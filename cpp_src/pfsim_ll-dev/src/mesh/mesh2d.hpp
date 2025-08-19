#ifndef SRC_MESH_MESH2D_HPP
#define SRC_MESH_MESH2D_HPP 1

// cache alignment macro
// should be equal to cacheline
// size or half of it
#ifndef CACHE_LINE_SIZE_DEFAULT
#define CACHE_LINE_SIZE_DEFAULT 64
#endif

#include <cstddef>
#include <cassert>
#include <src/aux/alignmem.hpp>

namespace mesh { namespace twd {

// class to manage storage orders
enum class StorageOrder2d{
    xy, /* row major */
    yx /* col major */
};


	
template <typename T>
class Mesh2d {
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
	Mesh2d(std::size_t m, std::size_t n, StorageOrder2d order=StorageOrder2d::xy) : 
		numRows(m), numCols(n), 
        nopmax(1),
		incRow(order==StorageOrder2d::xy? n*nopmax : nopmax), 
        incCol(order==StorageOrder2d::xy? nopmax : m*nopmax), 
        flat(true),
		mem(new T[aux::compute_aligned_size<T>(numRows*numCols*nopmax, CACHE_LINE_SIZE_DEFAULT)]()),
		data(aux::align_ptr(mem, CACHE_LINE_SIZE_DEFAULT))
	{
	}
	
	// specified buffer size, corresponds 
	// to adding a third dimension;
	Mesh2d(std::size_t m, std::size_t n, std::size_t localdim, StorageOrder2d order=StorageOrder2d::xy) : 
			numRows(m), numCols(n),
			nopmax(localdim), 
            incRow(order==StorageOrder2d::xy? n*nopmax : nopmax), 
            incCol(order==StorageOrder2d::xy? nopmax : m*nopmax),
			flat(localdim==1? true : false),
			mem(new T[aux::compute_aligned_size<T>(numRows*numCols*nopmax, CACHE_LINE_SIZE_DEFAULT)]()),
			data(aux::align_ptr(mem, CACHE_LINE_SIZE_DEFAULT))
	{
	}
	
	// standard destructor
	// to release memory
	~Mesh2d() 
	{
		delete[] mem;
	}
	
	void memset(T value) {
		for (std::size_t i=0; i<numRows*numCols*nopmax; ++i) {
			data[i] = value;
		}
	}
	
	// read-only access methods for 
	// order parameter values
	const T& operator() (std::size_t i, std::size_t j, std::size_t op) const 
	{
		//assert(!flat);
		assert(i<numRows && j<numCols);
		assert(op<nopmax);
		return data[i*incRow+j*incCol+op];
	}
	
	// write access methods
	T& operator() (std::size_t i, std::size_t j, std::size_t op) 
	{
		//assert(!flat);
		assert(i<numRows && j<numCols);
		assert(op<nopmax);
		return data[i*incRow+j*incCol+op];
	}
	
	// read-only access methods for 
	// flat mesh
	const T& operator() (std::size_t i, std::size_t j) const 
	{
		//assert(flat);
		assert(i<numRows && j<numCols);
		return data[i*incRow+j*incCol];
	}
	
	// write access methods
	T &operator() (std::size_t i, std::size_t j) 
	{
		//assert(flat);
		assert(i<numRows && j<numCols);
		return data[i*incRow+j*incCol];
	}

	
	// method to access pointer to data
	// might come in handy for some algorithms
	// but be sure to know what you do
	T* ptr() {
		return data;
	}
	
	T* memptr() {
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
	
	std::size_t opbufsize() const {
		return nopmax;
	}
	
	std::size_t xinc() const {
		return incRow;
	}
	
	std::size_t yinc() const {
		return incCol;
	}
	
	// delete copy constructor and assignment operator
	// so that no hidden errors can occur
	Mesh2d& operator=(const Mesh2d&) = delete;
    Mesh2d(const Mesh2d&) = delete;
	
private:
	// increments used for access
	// and mesh size
	std::size_t numRows, numCols;
	std::size_t nopmax;
	std::ptrdiff_t incRow, incCol;
	
	// 2 or 3 for 2D mesh or 3D mesh, might 
	// come in handy for MPI later
	//std::size_t dim;
	
	//attribute to check if it is "flat"
	bool flat;
	
	// pointer to data
	T* mem;
	T *data;
};

} /*namespace mesh*/ } /*namespace twd*/ 


#endif // SRC_MESH_MESH2D_HPP
