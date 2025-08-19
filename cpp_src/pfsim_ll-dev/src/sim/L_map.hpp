#ifndef SRC_SIM_LMAP_HPP
#define SRC_SIM_LMAP_HPP

#ifndef STEPVALUE
#define STEPVALUE 31.05
#endif

#ifndef LOWVAL
#define LOWVAL 0.18742
#endif

#ifndef HIVAL
#define HIVAL 0.65595
#endif

namespace sim {

template<typename T>
inline T L_map(T misorientation) {
	if (misorientation<=STEPVALUE) {
		return LOWVAL;
	} else {
		return HIVAL;
	}
}

} /*namespace sim*/

#endif // SRC_SIM_LMAP_HPP
