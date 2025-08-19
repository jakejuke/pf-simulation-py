#include <cstdio>
#include <cstddef>
#include <src/sim/genorientation.hpp>
#include <src/sim/calcmisorientation.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/lowtri.hpp>
#include <src/aux/savearray.hpp>

int main()  {

	std::size_t num = 45;
	aux::Buffer<double> buf(INCORI*num);

	// first generate some samples
	sim::gen_orientation(num, buf);

	// output the results
	/*
	for (std::size_t i=0; i<num; ++i) {
		for (std::size_t j=0; j<3; ++j) {
			for (std::size_t l=0; l<3; ++l) {
				std::printf("%lf ",buf.data[i*INCORI+3*j+l]);
			}
			std::printf("\n");
		}
		std::printf("\n");
	}
	*/

	// now calculate the misorientations
	// aux::Buffer<double> misbuf(num*(num-1)/2);
	aux::LowerTriangular<double> lowtribuf(num);
	sim::calc_all_misorientation(num, buf, lowtribuf);

	//print out the misorientation results
	for (std::size_t i=0; i<num; ++i) {
		for (std::size_t j=i+1; j<num; ++j) {
			std::printf("(%d, %d): %lf\n",i, j, lowtribuf(i,j));
		}
	}

	std::printf("\n");
}

