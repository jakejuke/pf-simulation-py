#include <src/aux/buffer.hpp>
#include <cstddef>
#include <cstdio>

int main() {

	std::size_t sz = 129;
	aux::Buffer<float> buf(sz);
	

	/*
	for (std::size_t i=0; i<buf.length; ++i) {
		std::printf("%d ",buf.data[i]);
	}
	std::printf("\nend of buffer\n");
	*/

	std::printf("computed length: %ld\n", aux::compute_aligned_size<decltype(buf.data[0])>(sz, 64));
	std::printf("mem: %p, mem mod 64: %ld\n",buf.mem,((std::size_t)buf.mem)%64);
	std::printf("data: %p, data mod 64: %ld\n",buf.data,((std::size_t)buf.data)%64);
	std::printf("dist: %ld\n", buf.data-buf.mem);
	
	/*
	double* p = new double[125];

	double* alignedp = p + static_cast<std::ptrdiff_t>(p)%64;
	std::printf("p alignment: %ld\n", static_cast<std::ptrdiff_t>(alignedp) % 64);

	delete[] p;
	*/
}


