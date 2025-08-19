#include <src/mesh/mesh2d.hpp>
#include <src/mesh/sumsq2d.hpp>
#include <cstdio>
#include <cstddef>


int main() {

	// initialize test case
	mesh::twd::Mesh2d<float> m(4,4,0);
	mesh::twd::Mesh2d<unsigned short> id(4,4,0);
	mesh::twd::Mesh2d<unsigned short> nop(4,4);
// 	for(std::size_t i=0; i<m.xsize(); ++i) {
// 		for(std::size_t j=0; j<m.ysize(); ++j) {
// 			m(i,j,0) = 0.5;
// 			id(i,j,0) = i*m.xsize()+j+1;
// 			nop(i,j) = 1;
// 		}
// 	}

	nop(1,1) += 2;
	m(1,1,0) = 0.5;
	id(1,1,0) = 1;
	m(1,1,1) = 0.8;
	id(1,1,1) = 2;

	//check for writing successful
	printf("posx	posy	val	id\n");
	for(std::size_t i=0; i<m.xsize(); ++i) {
		for(std::size_t j=0; j<m.ysize(); ++j) {
			printf("%ld, %ld, %f, %d\n",i, j, m(i,j,0), id(i,j,0));
		}
	}
	

	// create object to store the max values in
	mesh::twd::Mesh2d<float> l(4,4);

	mesh::twd::sumsq(nop, m, l);
	
	//print result to check for correctness
	printf("posx	posy	val	id\n");
	for(std::size_t i=0; i<m.xsize(); ++i) {
		for(std::size_t j=0; j<m.ysize(); ++j) {
			printf("%ld, %ld, %f\n",i, j, l(i,j));
		}
	}



	std::printf("some message\n");
}
