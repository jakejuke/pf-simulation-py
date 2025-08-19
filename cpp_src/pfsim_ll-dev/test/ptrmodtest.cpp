#include <iostream>
#include <cstddef>
#include <cstdint>
#include <cassert>

template <typename T>
inline void aligned_new(T** mem, T** aligned_mem, 
					  std::size_t length, std::size_t alignment) 
{
	assert(length>0 && alignment>0);
	// check if there is alreay sth stored
	if (*mem!=nullptr || *aligned_mem!=nullptr) {
		return;
	}
	
	// allocate memory
	*mem = new T[length+alignment-1];
	
	// check for success
	if (*mem==nullptr) return;
	
	// determine the next aligned element
	if (reinterpret_cast<std::uintptr_t>(*mem)%alignment == 0) {
		*aligned_mem = *mem;
	} else {
		*aligned_mem = reinterpret_cast<T*>(
					reinterpret_cast<char*>(*mem)
					+ alignment 
					- (reinterpret_cast<std::uintptr_t>(*mem) %
					alignment));
	}
}

int main() {

	std::size_t len = 400;
	std::size_t alignment = 8;
	float* mem=nullptr; float* data=nullptr;
	aligned_new(&mem, &data, len, alignment);
	/*
	mem = new double[len];

	// fix for the case that mem%64==0: data = mem;
	data = reinterpret_cast<double*>(reinterpret_cast<char*>(mem) + (64 - (reinterpret_cast<uintptr_t>(mem) % 64)));
	*/
	
	
	

	std::cout << "mem  value: " << mem << std::endl;
	std::cout << "mem mod 64: " << reinterpret_cast<std::size_t>(mem) % alignment << std::endl;
	std::cout << "data value: " << data << std::endl;
	std::cout << "data mod 64: " << reinterpret_cast<std::size_t>(data) % alignment << std::endl;
	std::cout << "dist: " << data - mem << std::endl;

	int **p = NULL;
	std::cout << "length of pointer: " << sizeof(p) << std::endl;


	delete[]  mem;
}
