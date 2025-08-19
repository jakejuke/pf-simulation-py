#include <src/sim2d.hpp>
#include <src/mesh2d.hpp>
#include <src/aux.hpp>
#include <cstddef>
#include <cstdio>

#ifndef NOF_GRAINS
#define NOF_GRAINS 656
#endif

double get_misorientation(float *misorientation, int grain_i, int grain_j) {
	float misori;

	if (grain_i==grain_j) {
		return 1;
	} else if (grain_i > grain_j) {
		index = grain_i*(grain_i-1)/2+grain_j;
	} else {
		index = grain_j*(grain_j-1)/2+grain_i;
	
	return misori[index];
}

// template <typename T, typename S>
// T get_Lcoeff_Cref(std::size_t n,
// 				T* misorientation,
// 				T* valbuf, std::ptrdiff_t incValbuf,
// 				S* idbuf, std::ptrdiff_t idbuf,
// 				T* coeffmaps, std::ptrdiff_t incCoeffmaps
//   ) {
// 	if(n>1) {
// 		
// 	} else {
// 		return c
// 	
// }
/*
void get_parameters(float misorientation, float *parameters, float *kVal, float *gamma, float *lVal) {
	int integer, rational;

	integer = trunc(misorientation);
	rational = round((misorientation - integer) * 100);

	*kVal = PARAMETERS(integer, rational, 1);
	*gamma = PARAMETERS(integer, rational, 2);
	*lVal = PARAMETERS(integer, rational, 3);
}*/


int main() { 

	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	auto elementsread = aux::load_array("../exec/misorientation", NOF_GRAINS, misorientation.data);

	const std::size_t angleres = 6800;
	aux::Buffer<float> coeffmaps(angleres*2);
	elementsread = sim::readcoeffmaps("../data/coeffmapsAniso/L2d.txt", coeffmaps.data, 2);
	elementsread = sim::readcoeffmaps("../data/coeffmapsAniso/gamma2d.txt", coeffmaps.data+1, 2);
	

	std::size_t n = 4;
	float valbuf[n] = {0.2, 0.3, 0.89, 0.93};
	std::size_t idbuf[n] = {1,2,4,5};

	for (std::size_t i=0; i<n; ++i) {
		std::printf("%f ",misorientation.data[i]);
	}
	std::printf("\n");

	float Lcoeff = sim::get_Lcoeff(n,
			misorientation.data,
			valbuf, 1,
			idbuf, 1,
			coeffmaps.data, 2);

	std::printf("result for L: %f\n", Lcoeff);

	std::size_t pos = 0;
	float gammacoeff = sim::get_Gammacoeff(n,
			pos,
			misorientation.data,
			valbuf, 1,
			idbuf, 1,
			coeffmaps.data, 2);

	std::printf("result for Gamma at pos %ld: %f\n", pos, gammacoeff);
}
