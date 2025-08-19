#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <src/sim/packnn2d.hpp>
#include <src/aux/ringproject.hpp>
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
	printf("posx	posy	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%ld	%ld	%f	%d	%d\n",i, j, val(i,j,0), id(i,j,0), nop(i,j));
		}
	}
	

	std::printf("return message: %d\n", linesread);

	// get buffer
	constexpr std::size_t len = LENGTH;
	unsigned short idbuf[len] = {0};
	
	for (std::size_t i=0; i<4; ++i) {
		for (std::size_t j=0; j<4; ++j) {
			float valbuf[5*len];
			for (std::size_t l=0; l<5*len; ++l) {
				valbuf[l] = 0;
			}

			std::size_t iprevious = aux::ringproject(i-1, 4);
			std::size_t inext = aux::ringproject(i+1, 4);
			std::size_t jprevious = aux::ringproject(j-1, 4);
			std::size_t jnext = aux::ringproject(j+1, 4);
			
			auto ret = sim::twd::packnn(i,j,iprevious,inext,
					jprevious,jnext,
					nop.ptr(), nop.xinc(), nop.yinc(),
					val.ptr(), val.xinc(), val.yinc(),
					id.ptr(), id.xinc(), id.yinc(),
					len,
					valbuf, 1,
					idbuf, 1);
			
			// print buffers
			std::printf("x, y, nop in nn: %ld, %ld, %d\n", i, j, ret);
			std::printf("id		sum of other\n");	
			for (std::size_t l=0; l<len; ++l) {
				std::printf("%d		%f\n",
						idbuf[l],valbuf[l]);
			}

		}
	}

	
}
