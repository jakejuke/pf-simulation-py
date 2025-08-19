#include <src/mesh/mesh2d.hpp>
#include <src/mesh/swap.hpp>
#include <cstdio>


int main() {

	mesh::twd::Mesh2d<float> m(4,4);
	m(1,1)=1; 
	m(1,1,0) = 3;

	mesh::twd::Mesh2d<float> k(4,4);
	k(1,1)=1; 
	k(1,1,0) = 5;

	// this is meant to produce an 
	// compiler error:
	// k = m;
	// and it does

	//likewise here
	//mesh::twd::Mesh2d<float> l(k);
	//and it does
	
	std::printf("some order parameter value: %f\n",m(1,1,0));

	std::printf("corresponding nof op !=0: %ld\n",m.get_localnop(1,1));

	std::printf("some message\n");
}
