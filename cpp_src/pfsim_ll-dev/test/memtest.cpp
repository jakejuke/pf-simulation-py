#include <src/mesh/mesh2d.hpp>
#include <iostream>
#include <cstddef>

template <typename T>
T* align_ptr(T* memptr, std::size_t desired_alignment) {
	std::size_t alignment = std::max(alignof(T), desired_alignment);
	return (T*) (((std::uintptr_t) memptr + alignment) & ~(alignment - 1 ));
}

template <typename T>
std::size_t compute_aligned_size(std::size_t length, std::size_t desired_alignment) {
	return length * sizeof(T) +
	std::max(desired_alignment, alignof(T));
}


int main() {
	
	std::size_t len = 145;
	std::size_t al = 64;
	double* mp = new double[compute_aligned_size<double>(len, al)];
	
	double* dp = align_ptr(mp, al);
	
	std::cout << "computed length: " << compute_aligned_size<double>(len, al) / sizeof(double) << std::endl;
	std::cout << "mp alignment: " << reinterpret_cast<std::uintptr_t>(mp) % al << std::endl;
	std::cout << "dp alignment: " << reinterpret_cast<std::uintptr_t>(dp) % al << std::endl;
	
	
	/*
	mesh::twd::Mesh2d<float> m(300,300);

	mesh::twd::Mesh2d<short> ids(300,300);

	std::printf("float size: %ld, short size %ld\n", sizeof(float), sizeof(short));
	std::printf("some message\n");
	*/
	
	delete[] mp;
}
