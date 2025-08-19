#ifndef SRC_AUX_RINGPROJECT_HPP
#define SRC_AUX_RINGPROJECT_HPP

namespace aux {

// helper function to determine
// the projection of a number onto
// the ring defined by the interval
// [0, ring_size-1];
// specifically int and not templated
// as this might cause more issues 
// than is serves for
inline std::size_t ringproject(int input, int ring_size) {
	return static_cast<std::size_t>((input+ring_size) % ring_size);
}

} /*namespace aux*/

#endif // SRC_AUX_RINGPROJECT_HPP
