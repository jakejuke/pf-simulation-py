#include <src/mesh/mesh2d.hpp>
#include <src/mesh/compat.hpp>
#include <src/mesh/twin.hpp>
#include <cstdio>


int main() {

	mesh::twd::Mesh2d<float> m(4,4);
	m(1,1,0) = 3;

	mesh::twd::Mesh2d<short> nop(4,4,1);
	nop(1,1) = 1;

	std::printf("some order parameter value: %f\n",m(1,1,0));

	std::printf("corresponding nof op !=0: %ld\n",nop(1,1));

	
	mesh::twd::Mesh2d<float> l(4,4,1);
	
	//this function apparently does not work
	//because it requires a copy constructor
	//auto t = mesh::twd::get_twin(m);


	//std::printf("compatibility test m and l: %d\n", mesh::twd:check_compat(m,l));
	//std::printf("ident test m and t: %d\n", mesh::twd::check_ident(m,t));
	

	std::printf("some message\n");
}
