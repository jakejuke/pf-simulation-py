#include <src/mesh/mesh2d.hpp>
#include <src/mesh/init2d.hpp>
#include <src/aux/buffer.hpp>
#include <src/sim/stepiso2d.hpp>
#include <src/mesh/swap.hpp>
#include <src/sim/findpos.hpp>
#include <cstdio>
#include <cstddef>

#define LENGTH 5

#ifndef NUM_ITER
#define NUM_ITER 3
#endif


int main() {

	std::size_t testm = 8;
	std::size_t testn = 8;
	// initialize test case
	mesh::twd::Mesh2d<float> val(testm,testn,0);
	mesh::twd::Mesh2d<unsigned short> id(testm,testn,0);
	mesh::twd::Mesh2d<unsigned short> nop(testm,testn);
	
	//read from file
	char filename[16] = "./testfileops3";
	auto linesread = mesh::twd::init_from_file(filename, nop, val, id);
	
	//check for writing successful
	/*
	printf("posx	posy	val		id	nop\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%ld	%ld	%f	%d	%d\n",i, j, val(i,j,0), id(i,j,0), nop(i,j));
		}
	}
	*/
	

	std::printf("return message: %d\n", linesread);
	/*
	printf("op1:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			auto currentpos = sim::findpos(nop(i,j),&id(i,j), (unsigned short)1);
			float currentval;
			if (currentpos == 2) {
				currentval = 0;
			} else {
				currentval = val(i,j,currentpos);
			}
			printf("%f ", currentval);
		}
		std::printf("\n");
	}
	printf("op2:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			auto currentpos = sim::findpos(nop(i,j),&id(i,j), (unsigned short)2);
			float currentval;
			if (currentpos == 2) {
				currentval = 0;
			} else {
				currentval = val(i,j,currentpos);
			}
			printf("%f ", currentval);
		}
		std::printf("\n");
	}
	
	printf("nop:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", nop(i,j));
		}
		std::printf("\n");
	}
	
	printf("id layer 0:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", id(i,j,0));
		}
		std::printf("\n");
	}
	printf("id layer 1:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", id(i,j,1));
		}
		std::printf("\n");
	}
	*/
	

	//create the meshes that we propagate to
	mesh::twd::Mesh2d<float> destval(testm,testn,0);
	mesh::twd::Mesh2d<unsigned short> destid(testm,testn,0);
	mesh::twd::Mesh2d<unsigned short> destnop(testm,testn);
	
	for (std::size_t l=0; l<NUM_ITER; ++l) {
		// perform propagation
		/*
		sim::twd::step(8,8, 
				0,0, 
				testm,testn, 
				nop.ptr(), nop.xinc(), nop.yinc(),
				val.ptr(), val.xinc(), val.yinc(),
				id.ptr(), id.xinc(), id.yinc(),
				destnop.ptr(), destnop.xinc(), destnop.yinc(),
				destval.ptr(), destval.xinc(), destval.yinc(),
				destid.ptr(), destid.xinc(), destid.yinc(),
				NOP_LOCAL_MAX,
				valbuf.data,
				idbuf.data);
		*/
		sim::twd::stepiso(nop, val, id, destnop, destval, destid);
		
		//check what is written in buffer
		/*
		for (std::size_t i=0; i<INCBUFF*NOP_LOCAL_MAX; ++i) {
			std::printf("%f ",valbuf.data[i]);
		}
		std::printf("\n");
		*/

		// output result
		printf("result of %ld iterations:\n",l+1);
		printf("op1:\n");
		for(std::size_t i=0; i<val.xsize(); ++i) {
			for(std::size_t j=0; j<val.ysize(); ++j) {
				auto currentpos = sim::findpos(destnop(i,j),&destid(i,j), (unsigned short) 1);
				float currentval;
				if (currentpos == 2) {
					currentval = 0;
				} else {
					currentval = destval(i,j,currentpos);
				}
				printf("%f ", currentval);
			}
			std::printf("\n");
		}
		printf("op2:\n");
		for(std::size_t i=0; i<val.xsize(); ++i) {
			for(std::size_t j=0; j<val.ysize(); ++j) {
				auto currentpos = sim::findpos(destnop(i,j),&destid(i,j), (unsigned short) 2);
				float currentval;
				if (currentpos == 2) {
					currentval = 0;
				} else {
					currentval = destval(i,j,currentpos);
				}
				printf("%f ", currentval);
			}
			std::printf("\n");
		}
		
		
		mesh::swap(val, destval);
		mesh::swap(nop, destnop);
		mesh::swap(id, destid);
		
	}
	
	/*
	printf("idbuf: %d %d\n",idbuf.data[0],idbuf.data[1]);
	
	printf("nop:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", destnop(i,j));
		}
		std::printf("\n");
	}
	
	printf("id layer 0:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", destid(i,j,0));
		}
		std::printf("\n");
	}
	printf("id layer 1:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", destid(i,j,1));
		}
		std::printf("\n");
	}
	printf("id layer 2:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", destid(i,j,1));
		}
		std::printf("\n");
	}
	printf("id layer 3:\n");
	for(std::size_t i=0; i<val.xsize(); ++i) {
		for(std::size_t j=0; j<val.ysize(); ++j) {
			printf("%d ", destid(i,j,1));
		}
		std::printf("\n");
	}
	*/

}
