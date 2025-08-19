#ifndef SRC_SIM_STEPISOPP3D_HPP
#define SRC_SIM_STEPISOPP3D_HPP 1

#include <cstddef>
#include <cstdio>
#include <src/mesh/mesh3d.hpp>
#include <src/mesh/compat3d.hpp>
#include <src/aux/ringproject.hpp>
#include <src/sim/packnn3d.hpp>

/* list of parameters to be specified during compile time:
 
#ifndef LCOEFF
#define LCOEFF 0.65595
#endif

#ifndef KAPPACOEFF
#define KAPPACOEFF 4.2686
#endif

#ifndef GAMMACOEFF
#define GAMMACOEFF 1.5
#endif

#ifndef MCOEFF
#define MCOEFF 8.5372
#endif

#ifndef DELTAT
#define DELTAT 0.010
#endif

#ifndef DELTAX 
#define DELTAX 0.5
#endif

#ifndef THRESHOLD
#define THRESHOLD 1e-5
#endif

*/

namespace sim { namespace thd {
	
// C-style implementation
template <typename T, typename S, typename R>
int stepisopp(std::size_t m, std::size_t n, std::size_t k,
			std::size_t maxnop,
			std::size_t izero, std::size_t jzero, std::size_t lzero,
			std::size_t mparent, std::size_t nparent, std::size_t kparent,
			S* orignop, 
			std::ptrdiff_t incRowOrignop, std::ptrdiff_t incColOrignop, std::ptrdiff_t incLayOrignop,
			T* origval, 
			std::ptrdiff_t incRowOrigval, std::ptrdiff_t incColOrigval, std::ptrdiff_t incLayOrigval,
			R* origid, 
			std::ptrdiff_t incRowOrigid, std::ptrdiff_t incColOrigid, std::ptrdiff_t incLayOrigid,
			S* destnop, 
			std::ptrdiff_t incRowDestnop, std::ptrdiff_t incColDestnop, std::ptrdiff_t incLayDestnop,
			T* destval, 
			std::ptrdiff_t incRowDestval, std::ptrdiff_t incColDestval, std::ptrdiff_t incLayDestval,
			R* destid, 
			std::ptrdiff_t incRowDestid, std::ptrdiff_t incColDestid, std::ptrdiff_t incLayDestid,
			const T deltat,
			T* pfield, std::ptrdiff_t incRowPfield, std::ptrdiff_t incColPfield, std::ptrdiff_t incLayPfield
			) {
			
	
	// process some parameters
	const T Lcoeff = LCOEFF;
	const T gammacoeff = GAMMACOEFF;
	const T kappacoeff = KAPPACOEFF;
	const T mcoeff = MCOEFF;
	const T deltax = DELTAX;
	
	int status = 0;
	
	
	// outer loops over each lattice site
	#pragma omp parallel for shared(status)
	for (std::size_t i=izero; i<m+izero; ++i) {
		
		if (status==-1) continue;
		
		/* determine surrounding positions
			* while taking into account the periodic
			* boundary conditions
			*/
		std::size_t iprevious = aux::ringproject(i-1,mparent);
		std::size_t inext = aux::ringproject(i+1,mparent);
		
		for (std::size_t j=jzero; j<n+jzero; ++j) {
			/* determine surrounding positions
			* while taking into account the periodic
			* boundary conditions
			*/
			std::size_t jprevious = aux::ringproject(j-1,nparent);
			std::size_t jnext = aux::ringproject(j+1,nparent);
			
			for (std::size_t l=lzero; l<k+lzero; ++l) {
				
				std::size_t lprevious = aux::ringproject(l-1,kparent);
				std::size_t lnext = aux::ringproject(l+1,kparent);
			
				// copy pointer to make it easier to work with
				// we will use these sections in the allocated
				// memory as buffer for future operations from now on;
				// region to store sum of surrouding values in
				T* valbufnn = &destval[i*incRowDestval+j*incColDestval+l*incLayDestval];
				// region where we have current values in
				T* valbuf = &origval[i*incRowOrigval+j*incColOrigval+l*incLayOrigval];
				// region where we keep ids
				S* idbuf = &destid[i*incRowDestid+j*incColDestid+l*incLayDestid];
				
		
				
				/* call packing function using the
					* buffers provided by pointer
					* and all the origs with corresponding
					* increments
					*/
				auto nop_to_propagate = sim::thd::packnn(i, j, l,
								iprevious, inext,
								jprevious, jnext,
								lprevious, lnext,
								orignop, 
								incRowOrignop, incColOrignop, incLayOrignop,
								origval, 
								incRowOrigval, incColOrigval, incLayOrigval,
								origid, 
								incRowOrigid, incColOrigid, incLayOrigid,
								maxnop,
								valbufnn, 1,
								idbuf, 1);
				// check overflow error
				if (nop_to_propagate>=maxnop) {
					#pragma omp atomic 
					status--;
					break;
				}
				

				/* actual number of nop that will 
				* later be stored, after values below
				* THREHOLD have been truncated
				*/
				S finalnop = nop_to_propagate;
				
				//some temporary varibales
				T laplacian;
				T fullsum = 0;
				T Gammasum;
				T currentresult;
				std::size_t currentpos = 0;
				
				// calculate the full sum of order 
				// parameters squared
				for (std::size_t nl=0; nl<orignop[i*incRowOrignop+j*incColOrignop+l*incLayOrignop]; ++nl) {
					fullsum += valbuf[nl]*valbuf[nl];
				}
				fullsum *= gammacoeff;
				
				// propagate first the values that were
				// already occupied 
				for (std::size_t nl=0; nl<orignop[i*incRowOrignop+j*incColOrignop+l*incLayOrignop]; ++nl) {
					
					// calculate laplacian term //
					laplacian = (valbufnn[nl] 
									- 6 * valbuf[nl])
								/ (deltax*deltax);
									
					// now subtract only the current
					// ids value from the complete square
					// sum 
					Gammasum = fullsum 
						- gammacoeff*valbuf[nl]*valbuf[nl];
						
					
					/* here comes the real formula 
					* that is derived from a simple 
					* Euler single step method;
					* see moelans2021, eq. (7)
					*/
					currentresult = valbuf[nl]
								- deltat * Lcoeff * (
												mcoeff * (
													valbuf[nl] * valbuf[nl] * valbuf[nl]
													- valbuf[nl]
													+ 2 * valbuf[nl] * Gammasum
													+ 2* valbuf[nl] * pfield[i*incRowPfield+j*incColPfield+l*incLayPfield]
												)
												- kappacoeff * laplacian
								);
								
					// truncate value below threshold
					if (currentresult < THRESHOLD) {
						// if it is below, then just decrement
						// final number of ops
						--finalnop;
					} else {
						// else store the variable and id;
						// operation on id here is either a copy 
						// to the same position in memory or
						// to a previous one, depending on if
						// a value has already been truncated before
						destid[i*incRowDestid+j*incColDestid+l*incLayDestid
								+currentpos] = idbuf[nl];
						destval[i*incRowDestval+j*incColDestval+l*incLayDestval+
						currentpos] = currentresult;
						++currentpos;
					}
				}
				
				// now propagate the rest
				for (std::size_t nl=orignop[i*incRowOrignop+j*incColOrignop+l*incLayOrignop]; nl<nop_to_propagate; ++nl) {
					// where we effectively only need 
					// to calculate the laplace term
					currentresult = deltat * Lcoeff * kappacoeff * valbufnn[nl] / (deltax*deltax);
					
					// and store only if it exceeds
					// the threshold
					if (currentresult < THRESHOLD) {
						--finalnop;
					} else {
						destid[i*incRowDestid+j*incColDestid+l*incLayDestid
								+currentpos] = idbuf[nl];
						destval[i*incRowDestval+j*incColDestval+l*incLayDestval+
						currentpos] = currentresult;
						++currentpos;
					}
				
				}
				
				// store the final number of non-zero order parameters
				destnop[i*incRowDestnop+j*incColDestnop+l*incLayDestnop] = finalnop;
			
				//check for status for immediate break
				if(status==-1) break;
			}
			//check for status for immediate break
			if(status==-1) break;
		}
	}
	return (status);
}


//wrapper function
template <typename T, typename S, typename R>
int stepisopp(mesh::thd::Mesh3d<S> &orignop, 
			mesh::thd::Mesh3d<T> &origval,
			mesh::thd::Mesh3d<R> &origid, 
			mesh::thd::Mesh3d<S> &destnop,
			mesh::thd::Mesh3d<T> &destval, 
			mesh::thd::Mesh3d<R> &destid,
			const T deltat,
			mesh::thd::Mesh3d<T> &pfield
			) {
		
		// assert compatibilities etc.*/
		int c=0;
		if(!mesh::thd::check_ident(orignop, destnop)) c++;
		if(!mesh::thd::check_ident(origval, destval)) c++;
		if(!mesh::thd::check_ident(origval, destval)) c++;
		if(c>0) {
			std::printf("error when checking for dimensions\n");
			return(-1);
		}
			
		return sim::thd::stepisopp(orignop.xsize(), orignop.ysize(), orignop.zsize(),
						origval.opbufsize(),
						(std::size_t) 0, (std::size_t) 0, (std::size_t) 0,
						orignop.xsize(), orignop.ysize(), orignop.zsize(),
						orignop.ptr(), 
						orignop.xinc(), orignop.yinc(), orignop.zinc(),
						origval.ptr(), 
						origval.xinc(), origval.yinc(), origval.zinc(),
						origid.ptr(), 
						origid.xinc(), origid.yinc(), origid.zinc(),
						destnop.ptr(), 
						destnop.xinc(), destnop.yinc(), destnop.zinc(),
						destval.ptr(), 
						destval.xinc(), destval.yinc(), destval.zinc(),
						destid.ptr(), 
						destid.xinc(), destid.yinc(), destid.zinc(),
						deltat,
						pfield.ptr(),
						pfield.xinc(), pfield.yinc(), pfield.zinc());
		
}		
	
} /*namespace thd*/ } /*namespace sim*/


#endif // SRC_SIM_STEPISO3D_HPP
