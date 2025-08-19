#include <cstdio>
#include <cstddef>
#include <src/sim/readcoeffmaps.hpp>


int main() {

	char filename[32] = "./gamma.txt";

	float *destination = new float[6800];

	sim::readcoeffmaps(filename, destination);

	for (std::size_t i=0; i<100; ++i ) {
		std::printf("%f ",destination[i]);
	}
	std::printf("\n");

}
