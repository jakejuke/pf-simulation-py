#ifndef SRC_MESH_TWIN_HPP
#define SRC_MESH_TWIN_HPP 1

#include <src/mesh/mesh2d.hpp>

namespace mesh { namespace twd {
	
	/* this function does not really
	 * work because at return the object
	 * is copied and the copy constructor 
	 * has been deleted
	 */
	
	template <typename T>
	Mesh2d<T> &get_twin(Mesh2d<T>& origin) {
		
		Mesh2d<T> ret(origin.xsize(), origin.ysize(), origin.opbufsize());
		
		return ret;
	}
	
} /*namespace twd*/ } /*namespace mesh*/

#endif // SRC_MESH_TWIN_HPP
