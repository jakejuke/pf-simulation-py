#include <src/mesh/mesh3d.hpp>
#include <src/mesh/init3d.hpp>
#include <src/sim/packnn3d.hpp>
#include <src/aux/ringproject.hpp>
#include <cstdio>
#include <cstddef>

#define LENGTH 5


int main() {

	// define system size
	const std::size_t m=4;
	const std::size_t n=4;
	const std::size_t k=4;
	
	// initialize test case
	mesh::thd::Mesh3d<float> val(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> id(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> nop(m,n,k);
	
	//read from file
	char filename[16] = "./testfileops5";
	auto linesread = mesh::thd::init_from_file(filename, nop, val, id);
	

	
	//check for reading successful
	printf("posx	posy	posz	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			for(std::size_t l=0; l<val.zsize(); ++l) {
				printf("%ld	%ld	%ld	%f	%d	%d\n",i, j, l, val(i,j,l,0), id(i,j,l,0), nop(i,j,l));
			}
		}
	}


	std::printf("return message: %d\n", linesread);

	// get buffer
	constexpr std::size_t len = LENGTH;
	unsigned short idbuf[len] = {0};

	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t l=0; l<k; ++l) {
				float valbuf[2*len];
				for (std::size_t l=0; l<2*len; ++l) {
					valbuf[l] = 0;
				}

				std::size_t iprevious = aux::ringproject(i-1, m);
				std::size_t inext = aux::ringproject(i+1, m);
				std::size_t jprevious = aux::ringproject(j-1, n);
				std::size_t jnext = aux::ringproject(j+1, n);
				std::size_t lprevious = aux::ringproject(l-1, k);
				std::size_t lnext = aux::ringproject(l+1, k);

				auto ret = sim::thd::packnn(i,j,l,
						iprevious,inext,
						jprevious,jnext,
						lprevious,lnext,
						nop.ptr(), nop.xinc(), nop.yinc(), nop.zinc(),
						val.ptr(), val.xinc(), val.yinc(), val.zinc(),
						id.ptr(), id.xinc(), id.yinc(), id.zinc(),
						len,
						valbuf, 1,
						idbuf, 1);

				// print buffers
				std::printf("x, y, z, nop in nn: %ld, %ld, %ld,  %d\n", i, j, l, ret);
				std::printf("id		sum of other\n");	
				for (std::size_t nl=0; nl<len; ++nl) {
					std::printf("%d		%f\n",
							idbuf[nl],valbuf[nl]);
				}
			}

		}
	}


}
