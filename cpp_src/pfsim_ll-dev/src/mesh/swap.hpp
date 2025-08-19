#ifndef SRC_MESH_SWAP_HPP
#define SRC_MESH_SWAP_HPP 1

#include <src/mesh/compat2d.hpp>

namespace mesh {

/* function to swap the pointer of two 
 * mesh objects; need to be of the same
 * type (i.e. 2d or 3d AND the data type 
 * e.g. short/float/double),
 * else results undefined behaviour;
 * check for types is implicitly done at
 * compile time, error is like:
 * "no matching function call"
*/
template <typename T>
void swap(T &a, T &b) {
	//check for compatibility first
	//TODO: make sure it works for 2d and 3d
	// assert(check_ident(a, b));
	
	//now swap the pointers, use
	//auto for type because type is
	//not known and compiler will fix;
	//also we have to swap both pointers,
	//memory as well as data pointer
	auto tmpptr = a.ptr();
	auto tmpmemptr = a.memptr();
	a.set_memptr(b.memptr(), b.ptr());
	b.set_memptr(tmpmemptr, tmpptr);
}

} /*namespace mesh*/

#endif //SRC_MESH_SWAP_HPP
