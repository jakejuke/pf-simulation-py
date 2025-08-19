#include <cstdlib>
#include <cstdio>

int main() {

	const std::size_t num = 16;
	auto buf = new double[num];

	for (std::size_t i=0; i<num; ++i) {
		//buf[i] = 0;
		std::printf("%lf ",buf[i]);
	}
	std::printf("\n");

	delete[] buf;
}
