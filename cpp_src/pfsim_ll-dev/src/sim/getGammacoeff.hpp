#ifndef SRC_SIM_GETGAMMACOEFF_HPP
#define SRC_SIM_GETGAMMACOEFF_HPP

#include <cstddef>

#ifndef GRAIN_ID_START
#define GRAIN_ID_START 1
#endif

namespace sim {

// procedure to obtain the value
// of the sum including gamma_ij
// for a particular lattice site
// for particular grain id
template <typename T, typename R>
T get_Gammacoeff(std::size_t n,
			std::size_t currentpos,
			T* misorientation,
			T* valbuf, std::ptrdiff_t incValbuf,
			R* idbuf,std::ptrdiff_t incIdbuf,
			T* coeffdata, std::ptrdiff_t incCoeffdata
  ) {
	
	// case: prefactor to the sum
	// is zero, so that we don't actually 
	// have to evaluate the sum
	if (currentpos >= n) {
		return 0;
	}

	// init local variables
	T qsum = 0;
	std::size_t index = 0;

	// loop over all the other ids
	auto ida = idbuf[currentpos*incIdbuf]; 
	for (std::size_t l=0; l<n; ++l) {

		auto idb = idbuf[l*incIdbuf];
		// if ids match, skip
		if (ida==idb) {
			continue;
		// generate index for
		} else if (idb>ida) {
			index = static_cast<std::size_t>(misorientation[(idb-GRAIN_ID_START)*(idb-1-GRAIN_ID_START)/2+ida-GRAIN_ID_START]*100-0.5);
		} else {
			index = static_cast<std::size_t>(misorientation[(ida-GRAIN_ID_START)*(ida-1-GRAIN_ID_START)/2+idb-GRAIN_ID_START]*100-0.5);
		}


		qsum += coeffdata[index*incCoeffdata]*valbuf[l*incValbuf]*valbuf[l*incValbuf];

	}

	return qsum;

}


} /*namespace sim*/

#endif // SRC_SIM_GETGAMMACOEFF_HPP
