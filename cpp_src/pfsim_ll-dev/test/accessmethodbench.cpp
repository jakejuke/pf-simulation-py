#include <cstdio>
#include <src/mesh/mesh2d.hpp>
#include <src/aux/walltime.hpp>


template <typename T>
void setvalue(mesh::twd::Mesh2d<T> &m, T value) {
	for (std::size_t i=0; i<m.xsize(); ++i) {
		for (std::size_t j=0; j<m.ysize(); ++j) {
			for (std::size_t op=0; op<m.opbufsize(); ++op) {
				m(i,j,op) = value;
			}
		}
	}
}

template <typename T>
void setvalue(std::size_t m, std::size_t n, std::size_t nop, 
		std::ptrdiff_t incRow, std::ptrdiff_t incCol,
		T* p, T value) {
	for (std::size_t i=0; i<m; ++i) {
		for (std::size_t j=0; j<n; ++j) {
			for (std::size_t op=0; op<nop; ++op) {
				p[i*incRow+j*incCol+nop] = value;
			}
		}
	}
	
}

int main() {

	src::aux::WallTime<double> timer;

	std::size_t m, n, nop;
	m=800; n=800; nop=64;
	mesh::twd::Mesh2d<double> wmesh(m,n,nop);

	double newvalue = 1;
	timer.tic();
	setvalue(wmesh, newvalue);
	double time_elapsed = timer.toc();

	std::printf("time elapsed with method access: %lf\n", time_elapsed);


	mesh::twd::Mesh2d<double> vmesh(m,n,nop);
	newvalue = 0;
	timer.tic();
	setvalue(m, n, nop, nop*n, nop, vmesh.ptr(), newvalue);
	time_elapsed = timer.toc();

	std::printf("time elapsed with direct access: %lf\n", time_elapsed);

}	




