#ifndef SRC_MESH_MESH2DVIEW_HPP
#define SRC_MESH_MESH2DVIEW_HPP 1

#include <cstddef>
#include <cassert>
#include <src/aux/ringproject.hpp>

namespace mesh { namespace twd {
	
	
template <typename T>
class Mesh2dView {
	/* general idea:
	 * object that contains ptr to data
	 * of a parent mesh and lets act
	 * subprograms on parts of the parent
	 * mesh only; this becomes useful when
	 * we really only want to propagate the 
	 * inner part of a mesh first using the
	 * single step method, and then do the 
	 * outer part later, when probably transmission
	 * between MPI processes is finished;
	 * also it will provide periodicity access
	 * features through one of it's methods;
	 * for implementation basically copy the 
	 * methods, inheritance is not an option
	 * because of possible ambiguities and
	 * problem with memory allocation as 
	 * constructors often also are directly
	 * inherited (or at least I don't know 
	 * about all the details)
	*/
	
	//TODO: implement mechanism that prevents incorrect
	// use for access of a "flat"/"unflat" mesh, rspctly

public:	
	// standard constructor;
	// will copy attributes from parent
	// mesh; usage is to pass parent mesh,
	// then where the view starts and then 
	// its size;
	Mesh2dView(Mesh2d<T> &parentmesh, std::ptrdiff_t xstart, std::ptrdiff_t ystart,
				std::size_t m, std::size_t n) : 
				nopmax(parentmesh.opbufsize()), numRows(m), numCols(n), 
				startRows(xstart), startCols(ystart),
				incRow(parentmesh.xinc()), incCol(parentmesh.yinc()),
				data(&parentmesh(xstart,ystart))
	{
	}
	
	//TODO: constructor to generate view of a slice
	// of a Mesh3d object
	
	// standard destructor
	// don't free any memory
	~Mesh2dView() 
	{
	}
	
	
	// read-only access methods for 
	// order parameter values
	// this time use int to account for 
	// negative indices that allow to incorporate
	// periodic boundary conditions
	const T& operator() (int i, int j, std::size_t op) const 
	{
		assert(op>=0 && op<nopmax);
		
		// addition on the corresponding ring
		// this is really dirty, TODO: come up with sth.
		// more efficient and more stable as this sol
		// does not handle values correctly that "leave
		// the lower end twice;
		int iaccess = aux::ringproject(i+startRows, parentNumRows);
		int jaccess = aux::ringproject(j+startCols, parentNumCols);
		return data[iaccess*incRow+jaccess*incCol+op];
	}
	
	// write access methods
	T& operator() (std::size_t i, std::size_t j, std::size_t op) 
	{
		assert(op>=0 && op<nopmax);
		
		int iaccess = aux::ringproject(i+startRows, parentNumRows);
		int jaccess = aux::ringproject(j+startCols, parentNumCols);
		return data[iaccess*incRow+jaccess*incCol+op];
	}
	
	// read-only access methods for 
	// flat mesh
	const T& operator() (std::size_t i, std::size_t j) const 
	{
		int iaccess = aux::ringproject(i+startRows, parentNumRows);
		int jaccess = aux::ringproject(j+startCols, parentNumCols);
		return data[iaccess*incRow+jaccess*incCol];
	}
	
	// write access methods
	T &operator() (std::size_t i, std::size_t j) 
	{
		int iaccess = aux::ringproject(i+startRows, parentNumRows);
		int jaccess = aux::ringproject(j+startCols, parentNumCols);
		return data[iaccess*incRow+jaccess*incCol];
	}

	// method to access pointer to data
	// might come in handy for some algorithms
	// but be sure to know what you do
	T* ptr() {
		return data;
	}
	
	// method to reset pointer, e.g. for swap
	void set_ptr(T *new_data) {
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
	
	// TODO: explicit definition of copy constructor
	// and assignment operator
	Mesh2dView& operator=(const Mesh2dView&) = delete;
    Mesh2dView(const Mesh2dView&) = delete;
	
private:
	// increments used for access
	// and mesh size
	std::size_t nopmax;
	std::size_t startRows, startCols;
	std::size_t numRows, numCols;
	std::size_t parentNumRows, parentNumCols;
	std::ptrdiff_t incRow, incCol;
	
	// 2 or 3 for 2D mesh or 3D mesh, might 
	// come in handy for MPI later
	//std::size_t dim;
	
	//attribute to check if it is "flat"
	//bool flat;
	
	// pointer to data
	T *data;
};

} /*namespace mesh*/ } /*namespace twd*/ 

#endif // SRC_MESH_MESH2DVIEW_HPP
