#include <cstdio>
#include <cstddef>
#include <src/sim/genorientation.hpp>
#include <src/sim/calcmisorientation.hpp>
#include <src/aux/buffer.hpp>
#include <src/aux/lowtri.hpp>

int main()  {

	std::size_t num = 3;
	aux::Buffer<float> buf(INCORI*num);

	// first generate some samples
	sim::gen_orientation(num, buf.data);
	for (std::size_t i=0; i<num; ++i) {
		for (std::size_t j=0; j<3; ++j) {
			for (std::size_t l=0; l<3; ++l) {
				std::printf("%lf ",buf.data[i*INCORI+3*j+l]);
			}
			std::printf("\n");
		}
		std::printf("\n");
	}

	// now calculate the misorientations
	float orientation_buffer[9];
	for (std::size_t i=GRAIN_ID_START; i<=num; ++i) {
		for (std::size_t j=GRAIN_ID_START; j<=num; ++j) {
			auto misorientation = sim::calc_misorientation(i,j,buf.data,orientation_buffer);

			std::printf("misorientation between grains %ld and %ld: %f\n",i,j,misorientation);
		}
	}

	// now use the function to calculate all misorientations
	auto misorientations = new float[num*(num-1)/2];
	sim::calc_all_misorientation(num,
								 buf.data,
								misorientations);
	
	std::printf("all together again:\n");
	for (std::size_t i=0; i<num; ++i) {
		for (std::size_t j=0; j<num; ++j) {
			if (i==j) {
				std::printf("misorientation between grains %ld and %ld: %f\n",i+GRAIN_ID_START,j+GRAIN_ID_START,1.);
				continue;
			}
			auto val = (j<i)? misorientations[i*(i-1)/2+j] : misorientations[j*(j-1)/2+i];
			std::printf("misorientation between grains %ld and %ld: %f\n",i+GRAIN_ID_START,j+GRAIN_ID_START,val);
		}
	}
	
	
	delete[] misorientations;
	
	// and the same again using the wrapper function for lowtri
	aux::LowerTriangular<float> lowtrimisorientation(num);
	sim::calc_all_misorientation(num,
							buf,
						 lowtrimisorientation);
	
	std::printf("all together again from lowtri:\n");
	for (std::size_t i=0; i<num; ++i) {
		for (std::size_t j=0; j<num; ++j) {
			auto val = (i==j)? 1 : lowtrimisorientation(i,j);
			std::printf("misorientation between grains %ld and %ld: %f\n",i+GRAIN_ID_START,j+GRAIN_ID_START,val);
		}
	}
							
							  


}

