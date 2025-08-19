#ifndef SRC_SIM_FINDPOS_HPP
#define SRC_SIM_FINDPOS_HPP 1

#include <cstddef>

namespace sim {
/* helper function to find if 
 * id already is available
 */
template <typename T>
std::size_t findpos(std::size_t length, T* buf, T id, std::size_t startidx) {
	
	// start from specified startidx
	// and walk backwards
	// note: decrement is done during check
	// for range, thus no third expression
	for (std::size_t k=startidx+1; k-->0; ) {
		if(buf[k]==id) {
			return k;
		}
	}
	
	// then start from startidx and
	// walk forward
	for (std::size_t k=startidx+1; k<length; ++k) {
		if(buf[k]==id) {
			return k;
		}
	} 
	
	return length;
}
	
} /*namespace sim*/

#endif
