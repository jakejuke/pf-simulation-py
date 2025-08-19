#ifndef SRC_AUX_CHECKSUCCESS_HPP
#define SRC_AUX_CHECKSUCCESS_HPP

#include <cstdlib>

namespace aux {
	
void check_success(int status) {
	if (status==-1 || status==0) {
		std::printf("data file input error: %d, aborting\n",status);
		std::exit (2);
	}
}
	
} /*namespace aux*/

#endif // SRC_AUX_CHECKSUCCESS_HPP
