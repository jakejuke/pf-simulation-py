#ifndef SRC_SIM_PACK2D_HPP
#define SRC_SIM_PACK2D_HPP 1

#include <cstddef>
#include <src/sim/findpos.hpp>

namespace sim { namespace twd {
	
template <typename T, typename S, typename R>
R pack(std::size_t i, std::size_t j,
				std::size_t iprevious, std::size_t inext,
				std::size_t jprevious, std::size_t jnext,
				const S* nop, std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop,
				const T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal,
				const R* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId,
				std::size_t bufsize,
				T* valbuf, std::ptrdiff_t incValbuf,
				R* idbuf) {
	
	
	
	/* copy values at current position 
		* into with stride of 5 valbuf
		*/
	/* copy the ids in the exact same order */
	S totalnop = 0;
	totalnop += nop[i*incRowNop+j*incColNop];
	for (std::size_t nl=0; nl<totalnop; ++nl) {
		valbuf[nl*incValbuf] = val[i*incRowVal+j*incColVal+nl];
		idbuf[nl] = id[i*incRowId+j*incColId+nl];
	}
		
	
	/* run over all the other positions
		* and check if the id already is available,
		* in case it is, save its pos;
		* and add the opvalue to
		* pos+1,
		* if id not available then add the id to
		* idbuf and store op value at corresponding 
		* position;
		* also keep track of numbers of order parameters
		* added!
		*/
	
	std::size_t coords[8] = {
		iprevious, j,
		inext, j,
		i, jprevious,
		i, jnext};
	
	for (std::size_t p=0; p<4; ++p) {
		
		for (std::size_t nl=0; nl<nop[coords[p*2]*incRowNop+coords[p*2+1]*incColNop]; ++nl) {
		
			// fetch id
			R currentid = id[coords[p*2]*incRowId+coords[p*2+1]*incColId+nl];
			
			// determine current ids position
			// in idbuf
			std::size_t idpos = sim::findpos(totalnop, idbuf, currentid, nl);
			
			// cover case when new id has to be added
			if (idpos==totalnop) {
				//check for overflow
				if (idpos >= bufsize) return (0);
				idbuf[idpos] = currentid;
				totalnop++;
			}
			
			//write the value to the valbuf
			valbuf[idpos*incValbuf+1] += val[coords[p*2]*incRowVal+coords[p*2+1]*incColVal+nl];
			
		}
	}
		
			
	/* we are now left with an array that contains 
		* all the ids that need to be propagated
		* and one that contains all values (surrounding
		* summed up alread); ready to propagate!
		*/
	
	return totalnop;
}
	
	
} /*namespace sim*/ } /*namespace twd*/


#endif // SRC_SIM_PACK2D_HPP
