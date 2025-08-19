#ifndef EXEC_CONFIG2D_HPP
#define EXEC_CONFIG2D_HPP
// this file is intended to provide information 
// about parameters to the compiler during compile
// time; just use #include <exec/config2d.hpp> in the
// file containing main()

//NOF_GRAINS of testcaseAniso=655, MAXIMAL GRAIN ID!!

#ifndef NOF_GRAINS
#define NOF_GRAINS 47
#endif

#ifndef LOCAL_NOP_MAX
#define LOCAL_NOP_MAX 32
#endif

#ifndef DELTAT
#define DELTAT 0.15 //0.010 default max 0.20
#endif

#ifndef LCOEFF
#define LCOEFF 0.806822
#endif

#ifndef KAPPACOEFF
#define KAPPACOEFF 0.381125
#endif

#ifndef GAMMACOEFF
#define GAMMACOEFF 1.5
#endif

#ifndef MCOEFF
#define MCOEFF 0.76225
#endif

#ifndef DELTAX 
#define DELTAX 0.5
#endif

#ifndef THRESHOLD
#define THRESHOLD 1e-5
#endif

//for pinning
#ifndef NOF_PARTICLES
#define NOF_PARTICLES 0
#endif

#ifndef PARTICLE_RADIUS
#define PARTICLE_RADIUS 2
#endif

#ifndef FIELD_EPSILON
#define FIELD_EPSILON 8.0
#endif

#endif // EXEC_CONFIG2D_HPP
























































































































































































































