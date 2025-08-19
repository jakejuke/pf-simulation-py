#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <cstdio>
#include <cstddef>

#define LENGTH 5


int main() {

	// initialize test case
	mesh::twd::Mesh2d<float> val(4,4,0);
	mesh::twd::Mesh2d<unsigned short> id(4,4,0);
	mesh::twd::Mesh2d<unsigned short> nop(4,4);
	
	//read from file
	char filename[16] = "./testfileops2";
	auto linesread = mesh::twd::init_from_file(filename, nop, val, id);
	
	//check for writing successful
	std::printf("first layer:\n");
	printf("posx	posy	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%ld	%ld	%f	%d	%d\n",i, j, val(i,j,0), id(i,j,0), nop(i,j));
		}
	}

	std::printf("second layer:\n");
	printf("posx	posy	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%ld	%ld	%f	%d	%d\n",i, j, val(i,j,1), id(i,j,1), nop(i,j));
		}
	}

	

	std::printf("return message: %ld\n", linesread);


}
