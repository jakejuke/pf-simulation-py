#include <src/mesh/mesh2d.hpp>
#include <src/mesh/mesh3d.hpp>
#include <src/mesh/swap.hpp>
#include <cstdio>


int main() {

	mesh::twd::Mesh2d<float> m(4,4);
	m(1,1,0) = 3;

	mesh::twd::Mesh2d<float> k(4,4);
	k(1,1,0) = 5;

	std::printf("m before swap: %p, k before swap: %p\n",m.ptr(),k.ptr());
	mesh::swap(m, k);
	std::printf("m after swap: %p, k after swap: %p\n",m.ptr(),k.ptr());

	mesh::thd::Mesh3d<float> u(4,4,4);
	u(1,1,1,0) = 3;

	mesh::thd::Mesh3d<float> v(4,4,4);
	v(1,1,1,0) = 5;

	std::printf("u before swap: %p, v before swap: %p\n",u.ptr(),v.ptr());
	mesh::swap(u, v);
	std::printf("u after swap: %p, v after swap: %p\n",u.ptr(),v.ptr());

	std::printf("some message\n");
}
