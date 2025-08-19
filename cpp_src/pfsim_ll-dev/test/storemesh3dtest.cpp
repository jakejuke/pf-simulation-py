#include <src/mesh/mesh3d.hpp>
#include <src/mesh/init3d.hpp>
#include <src/mesh/storemesh3d.hpp>
#include <src/mesh/maxop3d.hpp>
#include <src/aux/walltime.hpp>
#include <cstdio>
#include <cstddef>


int main() {

    #pragma omp parallel
    { 
        std::printf("hi from the threads\n");
    }
	
	std::size_t m,n,k;
	m=100; n=100; k=100;

	// initialize test case
	mesh::thd::Mesh3d<float> val(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> id(m,n,k,0);
	mesh::thd::Mesh3d<unsigned short> nop(m,n,k);
	

	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			for (std::size_t l=0; l<val.zsize(); ++l) {
				val(i,j,l,0) = 1.;
				id(i,j,l,0) = 1; 
				nop(i,j,l) += 1;
				val(i,j,l,1) = 2.;
				id(i,j,l,1) = 2; 
				nop(i,j,l) += 1;
			}
		}
	}

    aux::WallTime<float> wt;
	
	char outfile[32] = "./testfileops5outval";
    wt.tic();
	auto lineswritten = mesh::thd::storemesh(outfile, nop, val, id);
	std::printf("return message: %d in %fs\n", lineswritten,wt.toc());
	
    
	mesh::thd::Mesh3d<float> maxval(m,n,k);
	mesh::thd::Mesh3d<unsigned short> maxid(m,n,k);
	mesh::thd::maxop(nop, val, id, maxval, maxid);
	
	std::sprintf(outfile,"./testfileops5outmaxnop");
    wt.tic();
	lineswritten = mesh::thd::storemesh(outfile, maxval, maxid);
	std::printf("return message: %d in %fs\n", lineswritten,wt.toc());
	
	std::sprintf(outfile,"./testfileops5outmaxnopwoval");
    wt.tic();
	lineswritten = mesh::thd::storemesh(outfile, maxid);
	std::printf("return message: %d in %fs\n", lineswritten,wt.toc());


}
