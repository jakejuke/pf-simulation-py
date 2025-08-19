#include <cstdio>
#include <cstddef>
#include <src/aux/lowtri.hpp>

int main() {

	std::size_t dim = 4;

	aux::LowerTriangular<float> A(dim);

	float counter = 1;
	for (std::size_t i=1; i<dim; ++i) {
		for (std::size_t j=0; j<i; ++j) {
			A(i,j) = counter++;
		}
	}

	// print out the memory array
	for (std::size_t i=0; i<A.length; ++i) {
		std::printf("%f ",A.data[i]);
	}
	std::printf("\n");
}
