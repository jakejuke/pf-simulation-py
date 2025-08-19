#include <src/mesh/mesh3d.hpp>
#include <src/mesh/sumsq3d.hpp>
#include <cstdio>
#include <cstddef>


int main() {

	std::size_t m, n, k;
	m = 3; n = 3; k = 3;
	// initialize test case
	mesh::thd::Mesh3d<float> val(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> id(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> nop(m,n,k);
// 	for(std::size_t i=0; i<m.xsize(); ++i) {
// 		for(std::size_t j=0; j<m.ysize(); ++j) {
// 			m(i,j,0) = 0.5;
// 			id(i,j,0) = i*m.xsize()+j+1;
// 			nop(i,j) = 1;
// 		}
// 	}

	nop(1,1,1) += 2;
	val(1,1,1,0) = 0.5;
	id(1,1,1,0) = 1;
	val(1,1,1,1) = 0.8;
	id(1,1,1,1) = 2;

	//check for writing successful
	printf("posx	posy	val	id\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			for (std::size_t l=0; l<val.zsize(); ++l) {
				printf("%ld, %ld, %ld, %f, %d\n",i, j, l, val(i,j,l,0), id(i,j,l,0));
			}
		}
	}
	

	// create object to store the max values in
	mesh::thd::Mesh3d<float> opsq(m,n,k);

	mesh::thd::sumsq(nop, val, opsq);
	
	//print result to check for correctness
	printf("posx	posy	val	id\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			for (std::size_t l=0; l<val.zsize(); ++l) {
				printf("%ld, %ld, %ld, %f\n",i, j, l, opsq(i,j,l));
			}
		}
	}



	std::printf("some message\n");
}
