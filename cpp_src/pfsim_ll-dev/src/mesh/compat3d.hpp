#ifndef SRC_MESH_COMPAT3D_HPP
#define SRC_MESH_COMPAT3D_HPP 1


namespace mesh { namespace thd {

/* function to make sure that
 * mesh attributes match
*/
template <typename T, typename S>
bool check_compat(T &a, S &b) {
	
	bool c = true;
	if (a.xsize() != b.xsize()) c = false;
	if (a.ysize() != b.ysize()) c = false;
	if (a.zsize() != b.zsize()) c = false;
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
	if (a.zsize() != b.zsize()) c = false;
	if (a.opbufsize() != b.opbufsize()) c = false;
	return c;
}

} /*namespace thd */ } /*namespace mesh*/

#endif //SRC_MESH_COMPAT3D_HPP
