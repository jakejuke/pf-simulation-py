#ifndef SRC_SIM_PACKNN2D_HPP
#define SRC_SIM_PACKNN2D_HPP 1

#include <cstddef>
#include <src/sim/findpos.hpp>

namespace sim { namespace twd {
	
	
// function to look parse all nearest
// neighboring positions and add up values of
// the same id and store them in valbuf while
// storing all the ids in idbuf
template <typename T, typename S, typename R>
S packnn(std::size_t i, std::size_t j,
				std::size_t iprevious, std::size_t inext,
				std::size_t jprevious, std::size_t jnext,
				const S* nop, std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop,
				const T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
				const R* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId,
				std::size_t bufsize,
				T* valbuf, std::ptrdiff_t incValbuf,
				R* idbuf, std::ptrdiff_t incIdbuf
	  ) {
	
	
	
	// fetch number of ops nonzero
	// at current position (i,j)
	S totalnop = nop[i*incRowNop+j*incColNop];
	
	
	// zero out the valbuf while
	// copying ids to idbuf
	for (std::size_t nl=0; nl<totalnop; ++nl) {
		idbuf[nl*incIdbuf] = id[i*incRowId+j*incColId+nl];
		valbuf[incValbuf*nl] = 0;
	}
		
	
	// run over nearest neighboring positions
	// and check if id is available in idbuf already;
	// if it is, add the corresponding value
	// to the corresponding position in valbuf;
	// also keep track of ids by adding them
	// to idbuf
	std::size_t coords[8] = {
		iprevious, j,
		inext, j,
		i, jprevious,
		i, jnext};
	
	for (std::size_t p=0; p<4; ++p) {
		
		for (std::size_t nl=0; nl<nop[coords[p*2]*incRowNop+coords[p*2+1]*incColNop]; ++nl) {
		
			// fetch id in nearest neighbor
			R currentid = id[coords[p*2]*incRowId+coords[p*2+1]*incColId+nl];
			
			// determine current ids position
			// in idbuf
			std::size_t idpos = sim::findpos(totalnop, 
				idbuf,
 				currentid,
				nl);
			
			// cover case when new id has to be added
			if (idpos==totalnop) {
				//check for overflow
				if (idpos >= bufsize) return (0);
				idbuf[idpos] = currentid;
				
				// zero out additional position 
				valbuf[idpos] = 0;
				
				// increment counter
				totalnop++;
			}
			
			//write the value to the valbuf
			valbuf[idpos*incValbuf] += val[coords[p*2]*incRowVal+coords[p*2+1]*incColVal+nl];
			
		}
	}
		
			
	// valbuf now contains the sum of all nearest
	// neighbors while idbuf
	// has now also the ids of ops of nearest neighbors
	// that need to be propagated
	
	return totalnop;
}
	
	
} /*namespace sim*/ } /*namespace twd*/


#endif // SRC_SIM_PACK2D_HPP
