#include <src/mesh/mesh3d.hpp>
#include <src/mesh/init3d.hpp>
#include <src/mesh/storemesh3d.hpp>
#include <src/mesh/maxop3d.hpp>
#include <cstdio>
#include <cstddef>

#define LENGTH 5


int main() {

	const std::size_t m = 3;
	const std::size_t n = 3;
	const std::size_t k = 3;


	// initialize test case
	mesh::thd::Mesh3d<float> val(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> id(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> nop(m,n,k);
	
	//read from file
	char filename[16] = "./testfileops5";
	auto linesread = mesh::thd::init_from_file(filename, nop, val, id);
	
	//check for writing successful
	std::printf("first layer:\n");
	printf("posx	posy	posz	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			for (std::size_t l=0; l<val.zsize(); ++l) {
				printf("%ld	%ld	%ld	%f	%d	%d\n",i, j,l, val(i,j,l,0), id(i,j,l,0), nop(i,j,l));
			}
		}
	}

	std::printf("second layer:\n");
	printf("posx	posy	posz	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			for (std::size_t l=0; l<val.zsize(); ++l) {
				printf("%ld	%ld	%ld	%f	%d	%d\n",i, j,l, val(i,j,l,1), id(i,j,l,1), nop(i,j,l));
			}
		}
	}

	mesh::thd::Mesh3d<float> maxval(m,n,k);
	mesh::thd::Mesh3d<unsigned short> maxid(m,n,k);

	mesh::thd::maxop(nop, val, id, maxval, maxid);

	char outfile[32] = "./testfileops5maxnopout";
	mesh::thd::storemesh(outfile, maxval, maxid);
	

	std::printf("return message: %ld\n", linesread);


}
