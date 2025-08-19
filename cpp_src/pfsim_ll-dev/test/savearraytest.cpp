#include <cstddef>
#include <cstdio>
#include <src/aux/savearray.hpp>
#include <src/aux/loadarray.hpp>
#include <src/aux/lowtri.hpp>


int main() {

	std::size_t num = 100;
	auto* buf = new float[num];

	for (std::size_t i=0; i<num; ++i) {
		buf[i] = i+1;
	}

	char filename[32] = "./testfile";
	aux::save_array(filename, num, buf);
	

	aux::load_array(filename, num, buf);

	
	for (std::size_t i=0; i<num; ++i) {
		std::printf("%f ",buf[i]);
	}
	std::printf("\n");

	aux::LowerTriangular<double> tri(num);


	aux::save_array(filename, tri.length, tri.data);
	

	aux::load_array(filename, tri.length, tri.data);

	for (std::size_t i=0; i<tri.length; ++i) {
		tri.data[i]++;
	}

	delete[] buf;
}
