#include <src/sim2d.hpp>
#include <src/mesh2d.hpp>
#include <src/aux.hpp>
#include <cstddef>
#include <cstdio>

#ifndef NOF_GRAINS
#define NOF_GRAINS 655
#endif


int main() { 

	aux::LowerTriangular<float> misorientation(NOF_GRAINS);
	auto elementsread = aux::load_array("../exec/misorientation", NOF_GRAINS, misorientation.data);

	const std::size_t angleres = 6800;
	aux::Buffer<float> coeffmaps(angleres*2);
	elementsread = sim::readcoeffmaps("../data/coeffmapsAniso/Aniso_2D/L2d_min0001.txt", coeffmaps.data, 2);
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
