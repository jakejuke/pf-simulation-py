#include <src/sim/saveorientation.hpp>

int main() {

	std::size_t len = 3;

	double buf[len*INCORI];

	for (std::size_t i=0; i<len*INCORI; ++i) {
		buf[i] = i+1;
	}

	char filename[32] = "./orimap.txt";
	auto lineswritten = sim::save_orientations(filename, len, buf);

	std::printf("lines written: %d\n",lineswritten);

}
