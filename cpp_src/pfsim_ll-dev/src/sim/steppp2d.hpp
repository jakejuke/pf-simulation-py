#ifndef SRC_SIM_STEPPP2D_HPP
#define SRC_SIM_STEPPP2D_HPP 1

#include <src/mesh/mesh2d.hpp>
#include <src/aux/ringproject.hpp>
#include <src/sim/packnn2d.hpp>
#include <src/sim/getLcoeff.hpp>
#include <src/sim/getGammacoeff.hpp>
#include <cstddef>

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

#ifndef DELTAX 
#define DELTAX 0.5
#endif

#ifndef THRESHOLD
#define THRESHOLD 1e-5
#endif

*/

namespace sim { namespace twd {
	
// C-style implementation
template <typename T, typename S, typename R>
int steppp(std::size_t m, std::size_t n, std::size_t maxnop,
			std::size_t izero, std::size_t jzero,
			std::size_t mparent, std::size_t nparent,
			S* orignop, std::ptrdiff_t incRowOrignop, std::ptrdiff_t incColOrignop,
			T* origval, std::ptrdiff_t incRowOrigval, std::ptrdiff_t incColOrigval,
			R* origid, std::ptrdiff_t incRowOrigid, std::ptrdiff_t incColOrigid,
			S* destnop, std::ptrdiff_t incRowDestnop, std::ptrdiff_t incColDestnop,
			T* destval, std::ptrdiff_t incRowDestval, std::ptrdiff_t incColDestval,
			R* destid, std::ptrdiff_t incRowDestid, std::ptrdiff_t incColDestid,
			const T deltat,
			// particle field input
			T* pfield, std::ptrdiff_t incRowPfield, std::ptrdiff_t incColPfield,
			// input of misorientations
			T* misorientation,
			// input of coefficient maps
			T* coeffdata, std::ptrdiff_t incCoeff) {
			
	
	// initialize some parameters
	const T kappacoeff = KAPPACOEFF;
	const T mcoeff = MCOEFF;
	const T deltax = DELTAX;
	
	int status = 0;
	
	
	// outer loops over each lattice site
	#pragma omp parallel for shared(status)
	for (std::size_t i=izero; i<m+izero; ++i) {
		
		// mechanism to stop calculation
		// if buffer overflow occurs
		if(status==-1) continue;
		
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
			
			// copy pointer to make it easier to work with
			// we will use these sections in the allocated
			// memory as buffer for future operations from now on;
			// region to store sum of surrouding values in
			T* valbufnn = &destval[i*incRowDestval+j*incColDestval];
			// region where we have current values in
			T* valbuf = &origval[i*incRowOrigval+j*incColOrigval];
			// region where we keep ids
			S* idbuf = &destid[i*incRowDestid+j*incColDestid];
	
			
			/* call packing function using the
				* buffers provided by pointer
				* and all the origs with corresponding
				* increments
				*/
			auto nop_to_propagate = sim::twd::packnn(i, j,
							iprevious, inext,
							jprevious, jnext,
							orignop, incRowOrignop, incColOrignop,
							origval, incRowOrigval, incColOrigval,
							origid, incRowOrigid, incColOrigid,
							maxnop,
							valbufnn, 1,
							idbuf, 1);
			
			// check for buffer overflow conditions
			if (nop_to_propagate==0) {
				#pragma omp atomic
				status--;
				break;
			}
			
			
			// procedure to obtain L specific for each
			// lattice site; pass here the nop at current
			// lattice site, as the values for the additional
			// surrounding ids evaluate to zero and will not 
			// contribute
			T Lcoeff = sim::get_Lcoeff(
									orignop[i*incRowOrignop+j*incColOrignop],
									misorientation,
									valbuf, 1,
									idbuf, 1,
									coeffdata, incCoeff);

			
			/* actual number of nop that will 
			 * later be stored, after values below
			 * THRESHOLD have been discarded
			 */
			S finalnop = nop_to_propagate;
			
			
			//some temporary varibales
			T laplacian;
			T currentresult;
			std::size_t currentpos = 0;
			/* propagate each order parameter
				* individually that alread is nonzero, 
				* store result directly
				* into destval;
				*/
			for (std::size_t l=0; l<orignop[i*incRowOrignop+j*incColOrignop]; ++l) {
				
				/* calculate laplacian term */
				laplacian = (valbufnn[l] 
								-4*valbuf[l]
							) 
							/ (deltax*deltax);
								
				// generate gamma, includes the sum,
				// similar to before pass nop here,
				// values for newly added ids do not 
				// contribute; case l>=orignop is
				// handled inside the function
				T Gammalocal = sim::get_Gammacoeff(
					orignop[i*incRowOrignop+j*incColOrignop],
					l,
					misorientation,
					valbuf, 1,
					idbuf, 1,
					coeffdata+1, incCoeff);
					
				
				/* here comes the real formula 
				 * that is derived from a simple 
				 * Euler single step method;
				 * see moelans2021, eq. (7)
				 */
				currentresult = valbuf[l]
							- deltat * Lcoeff * (
											mcoeff * (
												valbuf[l] 
												* valbuf[l] 
												* valbuf[l]
												- valbuf[l]
												+ 2 * valbuf[l] * Gammalocal
												+ 2 * valbuf[l] * pfield[i*incRowPfield+j*incColPfield]
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
					destid[i*incRowDestid+j*incColDestid+currentpos] = idbuf[l];
					destval[i*incRowDestval+j*incColDestval+currentpos] = currentresult;
					++currentpos;
				}
				
				
			}
			
			// now propagate the rest, i.e. order parameters
			// that have been zero before
			for (std::size_t l=orignop[i*incRowOrignop+j*incColOrignop]; l<nop_to_propagate; ++l) {
				// where we effectively only need 
				// to calculate the laplace term
				currentresult = deltat * Lcoeff * kappacoeff * valbufnn[l] / (deltax*deltax);
				
				// and store it only if it exceeds the THRESHOLD
				if (currentresult < THRESHOLD) {
					--finalnop;
				} else {
					destid[i*incRowDestid+j*incColDestid+currentpos] = idbuf[l];
					destval[i*incRowDestval+j*incColDestval+currentpos] = currentresult;
					++currentpos;
				}
			}
			
			// store the final number of non-zero order parameters
			destnop[i*incRowDestnop+j*incColDestnop] = finalnop;
			
			//check for status for immediate break
			if(status==-1) break;
		}
		
	}
	return (status);
}


//wrapper function
template <typename T, typename S, typename R>
int steppp(mesh::twd::Mesh2d<S> &orignop, 
			mesh::twd::Mesh2d<T> &origval,
			mesh::twd::Mesh2d<R> &origid, 
			mesh::twd::Mesh2d<S> &destnop,
			mesh::twd::Mesh2d<T> &destval, 
			mesh::twd::Mesh2d<R> &destid,
			const T deltat,
			mesh::twd::Mesh2d<T> &pfield,
			T* misorientation,
			T* coeffdata, std::ptrdiff_t incCoeff) {
		
		// TODO: assert compatibilities etc.*/
			
		return sim::twd::steppp(orignop.xsize(), orignop.ysize(), origval.opbufsize(),
						(std::size_t) 0, (std::size_t) 0,
						orignop.xsize(), orignop.ysize(),
						orignop.ptr(), orignop.xinc(), orignop.yinc(), 
						origval.ptr(), origval.xinc(), origval.yinc(),
						origid.ptr(), origid.xinc(), origid.yinc(),
						destnop.ptr(), destnop.xinc(), destnop.yinc(),
						destval.ptr(), destval.xinc(), destval.yinc(),
						destid.ptr(), destid.xinc(), destid.yinc(),
						deltat,
						pfield.ptr(), pfield.xinc(), pfield.yinc(),
						misorientation,
						coeffdata, incCoeff);
		
}		
	
} /*namespace twd*/ } /*namespace sim*/


#endif // SRC_SIM_STEP_HPP
