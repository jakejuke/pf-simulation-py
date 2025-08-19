#include <src/mesh/mesh3d.hpp>
#include <src/mesh/compat3d.hpp>
#include <cstdio>


int main() {

	mesh::thd::Mesh3d<float> m(4,4,4,64);
	m(1,1,1,0) = 3;
	m(0,0,0,0) = 666;

	mesh::thd::Mesh3d<short> nop(4,4,4);
	nop(1,1,1) = 1;

	std::printf("some order parameter value: %f\n",m(1,1,1,0));
	std::printf("some order parameter value: %f\n",m(0,0,0,0));

	//std::printf("corresponding nof op !=0: %ld\n",nop(1,1,1));
	
	std::printf("assert compatibility: %d\n", (int) mesh::thd::check_compat(m,nop));
	
	mesh::thd::Mesh3d<float> k(4,4,4);
	k(0,0,0) = 666;
	k(1,1,1) = 3;
	std::printf("some order parameter value: %f\n",k(1,1,1,0));
	std::printf("some order parameter value: %f\n",k(0,0,0,0));

	
// 	mesh::twd::Mesh3d<float> l(4,4,1);
	std::size_t localdim = 0;
	localdim = localdim==0? NOP_LOCAL_MAX : localdim;
	std::printf("localdim: %d\n", localdim);
	std::printf("number: %d\n", NOP_LOCAL_MAX);
	
	std::printf("some message\n");
}
