#ifndef SRC_SIM_PACK3D_HPP
#define SRC_SIM_PACK3D_HPP 1

#include <cstddef>
#include <cstdio>
#include <src/sim/findpos.hpp>

namespace sim { namespace thd {
	
template <typename T, typename S, typename R>
S pack(std::size_t i, std::size_t j, std::size_t l,
		std::size_t iprevious, std::size_t inext,
		std::size_t jprevious, std::size_t jnext,
		std::size_t lprevious, std::size_t lnext,
		const S* nop, std::ptrdiff_t incRowNop, std::ptrdiff_t incColNop, std::ptrdiff_t incLayNop,
		const T* val, std::ptrdiff_t incRowVal, std::ptrdiff_t incColVal, std::ptrdiff_t incLayVal,
		const R* id, std::ptrdiff_t incRowId, std::ptrdiff_t incColId, std::ptrdiff_t incLayId,
		std::size_t bufsize,
		T* valbuf, std::ptrdiff_t incValbuf,
		R* idbuf) {
	
	
	
	/* copy the ids in the exact same order */
	S totalnop = 0;
	totalnop += nop[i*incRowNop+j*incColNop+l*incLayNop];
	for (std::size_t nl=0; nl<totalnop; ++nl) {
		valbuf[nl*incValbuf] = val[i*incRowVal+j*incColVal+l*incLayVal+nl];
		idbuf[nl] = id[i*incRowId+j*incColId+l*incLayId+nl];
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
	
	std::size_t coords[18] = {
		iprevious, j, l,
		inext, j, l,
		i, jprevious, l,
		i, jnext, l,
		i, j, lprevious,
		i, j, lnext};
	
	for (std::size_t p=0; p<6; ++p) {
		
		for (std::size_t nl=0; nl<nop[coords[p*3]*incRowNop + coords[p*3+1]*incColNop + coords[p*3+2]*incLayNop]; ++nl) {
		
			// fetch id
			R currentid = id[coords[p*3]*incRowId
							+coords[p*3+1]*incColId
							+coords[p*3+2]*incLayId 
							+nl];
			
			// determine current ids position
			// in idbuf
			auto idpos = sim::findpos(totalnop, idbuf, currentid, nl);
			
			// cover case when new id has to be added
			if (idpos==totalnop) {
				//check for overflow
				if (idpos >= bufsize) return (0);
				idbuf[idpos] = currentid;
				totalnop++;
			}
			
			//write the value to the valbuf
			valbuf[idpos*incValbuf+1] += val[coords[p*3]*incRowVal
											+coords[p*3+1]*incColVal
											+coords[p*3+2]*incLayVal
											+nl];
			
		}
	}
		
			
	/* we are now left with an array that contains 
		* all the ids that need to be propagated
		* and one that contains all values (surrounding
		* summed up alread); ready to propagate!
		*/
	
	return totalnop;
}
	
	
} /*namespace thd*/ } /*namespace sim*/


#endif // SRC_SIM_PACK3D_HPP
