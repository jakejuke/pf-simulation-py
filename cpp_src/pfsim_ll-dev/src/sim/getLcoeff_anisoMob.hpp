#ifndef SRC_SIM_GETLCOEFF_ANISOMOB_HPP
#define SRC_SIM_GETLCOEFF_ANISOMOB_HPP 1

#include <cstddef>
#include <algorithm>
#include <exec/config3d.hpp>

#ifndef GRAIN_ID_START
#define GRAIN_ID_START 1
#endif

#ifndef LCOEFF
#define LCOEFF 0.65595
#endif

namespace sim {

// procedure to obtain L specific for each
// lattice site, takes packed buffers as
// argument
template <typename T, typename R>
T get_Lcoeff_anisoMob(std::size_t n,
			 T* misorientation,
			 T* valbuf, std::ptrdiff_t incValbuf,
			 R* idbuf, std::ptrdiff_t incIdbuf,
			 T* coeffdata, std::ptrdiff_t incCoeffdata
	) {
	
	
	
	// other procedure
	T Lcoeff = 0;
	T denominator = 0;
	T opproduct = 0;
	
	std::size_t index = 0;
	
	// check for case where n=1:
	// here there will only be a single contrib
	// to the sum, Lcoeff will evaluate to zero
	// and we will return a default value here
	if (n==1) {
		index = 30*100;
		return coeffdata[incCoeffdata*index];
	}
		
	// loop over grain ids provided
	for (std::size_t l=0; l<n; ++l) {
		for (std::size_t nl=l+1; nl<n; ++nl) {
			
			// fetch ids incorrect ordering as required
			// to access lower triangular matrix
			auto ida = std::max(idbuf[l*incIdbuf], idbuf[nl*incIdbuf]);
			auto idb = std::min(idbuf[l*incIdbuf], idbuf[nl*incIdbuf]);
			
			
			// generate index from misorientation,
			// maps have values for misorientations
			// in [0,68], stepsize 0.1
			index = static_cast<std::size_t>(
				misorientation[(ida-GRAIN_ID_START)
				*(ida-1-GRAIN_ID_START)/2
				+idb-GRAIN_ID_START]*100-1);
				
			// anisotropic mobility for abnormal grain (one grain is growing MOBRATION times faster?)
			if(ida == ABNORMAL_GRAIN || idb == ABNORMAL_GRAIN) {
				index = index*MOBRATIO;
			}

			//procedure for obtaining L
			opproduct = valbuf[incValbuf*l]
						* valbuf[incValbuf*l]
						* valbuf[incValbuf*nl]
						* valbuf[incValbuf*nl];
						
			Lcoeff += opproduct*coeffdata[incCoeffdata*index];
			
			denominator += opproduct;
		}
	}
	
	// handle exception, when there are
	// multiple ops to propagate but at 
	// the current lattice site there is 
	// only one of them nonzero; usually
	// the case only in the very beginning
	// of the simulation;
	// then return default value;
	// TODO: catch this exception earlier
	if (denominator==0) {
		index = 30*100;
		return coeffdata[incCoeffdata*index];
	}
	return Lcoeff/=denominator;
}
	
} /*namespace sim*/


#endif // SRC_SIM_GETLCOEFF_HPP
