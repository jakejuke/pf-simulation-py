#ifndef SRC_SIM_GAMMAMAP_HPP
#define SRC_SIM_GAMMAMAP_HPP

#ifndef STEPVALUE
#define STEPVALUE 31.05
#endif

#ifndef LOWVAL
#define LOWVAL 0.54402
#endif

#ifndef HIVAL
#define HIVAL 1.5
#endif

namespace sim {

template<typename T>
inline T gamma_map(T misorientation) {
	if (misorientation<=STEPVALUE) {
		return LOWVAL;
	} else {
		return HIVAL;
	}
}

} /*namespace sim*/

#endif //SRC_SIM_GAMMAMAP_HPP
