#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <cstdio>
#include <iostream>
#include <cstddef>

#define LENGTH 5


int main() {

	// initialize test case
	std::size_t m,n;
	m = n = 40;
	float epsilon = 3.0;
	mesh::twd::Mesh2d<float> field(m,n);
	
	//read from file
	char filename[] = "./particlefield.txt";
	auto linesread = mesh::twd::init_from_file(filename, field, epsilon);
	
	// output field
	for (std::size_t i=0; i<field.xsize(); ++i) {
		for (std::size_t j=0; j<field.ysize(); ++j) {
			std::cout << field(i,j) << " ";
		}
		std::cout << std::endl;
	}
	

	std::printf("return message: %d\n", linesread);


}
