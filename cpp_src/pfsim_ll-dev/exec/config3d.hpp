#ifndef EXEC_CONFIG3D_HPP
#define EXEC_CONFIG3D_HPP
// this file is intended to provide information
// about parameters to the compiler during compile
// time; just use #include <exec/config2d.hpp> in the
// file containing main()

#ifndef LOCAL_NOP_MAX
#define LOCAL_NOP_MAX 96
#endif

//MAXIMUM ID NOT NUMBER OF GRAINS
#ifndef NOF_GRAINS
#define NOF_GRAINS 39
#define NOF_GRAINS 39
#endif

#ifndef DELTAT
#define DELTAT 0.010
#endif

#ifndef LCOEFF
#define LCOEFF 9.83929
// LCOEFF 0.65595 // sigma = 2.8, iso sigma_init = sigma
#endif

#ifndef KAPPACOEFF
#define KAPPACOEFF 0.381125
// KAPPACOEFF 0.38112 // aniso, 2.8
// KAPPACOEFF 2.7441 // sigma = 1.8
// KAPPACOEFF 1.9818 // sigma = 1.3
// KAPPACOEFF 1.2196 // sigma = 0.8
#endif

#ifndef GAMMACOEFF
#define GAMMACOEFF 1.5
// GAMMACOEFF 1.5 // sigma = 2.8, iso sigma_init = sigma
#endif

#ifndef MCOEFF
#define MCOEFF 0.76225
// MCOEFF 0.76225 // aniso, 2.8
// MCOEFF 5.4882 // sigma = 1.8
// MCOEFF 3.9637 // sigma = 1.3
// MCOEFF 2.4392 // sigma = 0.8
#endif

#ifndef DELTAX
#define DELTAX 0.5
#endif

#ifndef THRESHOLD
#define THRESHOLD 1e-5
#endif

#ifndef NOF_PARTICLES
#define NOF_PARTICLES 0
// NOF_PARTICLES 10315
#endif

#ifndef PARTICLE_RADIUS
#define PARTICLE_RADIUS 3
#endif

#ifndef FIELD_EPSILON
#define FIELD_EPSILON 4.0
#endif

#ifndef MOBRATIO
#define MOBRATIO 100
#endif

#ifndef ABNORMAL_GRAIN
#define ABNORMAL_GRAIN 1559
#endif

#ifndef L_SINGLE_GRAIN
//#define L_SINGLE_GRAIN 0.18742 // sigma = 0.8, iso sigma_init = sigma
#define L_SINGLE_GRAIN 0.30455 // sigma = 1.3, iso sigma_init = sigma
//#define L_SINGLE_GRAIN 0.42168 // sigma = 1.8, iso sigma_init = sigma
//#define L_SINGLE_GRAIN 0.65595 // sigma = 2.8, iso sigma_init = sigma
#endif

#ifndef GAMMA_SINGLE_GRAIN
//#define GAMMA_SINGLE_GRAIN 0.54402 // sigma = 0.8, iso sigma_init = sigma
#define GAMMA_SINGLE_GRAIN 0.60993 // sigma = 1.3, iso sigma_init = sigma
//#define GAMMA_SINGLE_GRAIN 0.74182 // sigma = 1.8, iso sigma_init = sigma
//#define GAMMA_SINGLE_GRAIN 1.5 // sigma = 2.8, iso sigma_init = sigma
#endif

#endif // EXEC_CONFIG2D_HPP

////////////////////////////
// Nothing past this line //
////////////////////////////


