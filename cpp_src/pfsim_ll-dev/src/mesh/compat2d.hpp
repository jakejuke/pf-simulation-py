#ifndef SRC_MESH_COMPAT2D_HPP
#define SRC_MESH_COMPAT2D_HPP 1


namespace mesh { namespace twd {

/* function to make sure that
 * mesh attributes match
*/
template <typename T, typename S>
bool check_compat(T &a, S &b) {
	
	bool c = true;
	if (a.xsize() != b.xsize()) c = false;
	if (a.ysize() != b.ysize()) c = false;
	return c;
}

/* function to check for meshes to
 * be identical regarding attributes,
 * might come in handy if we want to 
 * copy one
*/
template <typename T, typename S>
bool check_ident(T &a, S &b) {
	
	bool c = true;
	if (a.xsize() != b.xsize()) c = false;
	if (a.ysize() != b.ysize()) c = false;
	if (a.opbufsize() != b.opbufsize()) c = false;
	return c;
}

} /*namespace twd */ } /*namespace mesh*/

#endif //SRC_MESH_SWAP_HPP
