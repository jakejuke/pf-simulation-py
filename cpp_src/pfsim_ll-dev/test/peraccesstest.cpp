#include <cstdio>
#include <cstddef>
#include <src/aux/ringproject.hpp>


int main() {

	int m;	
	int pos;

	for (;;)  {
		std::scanf("%d %d", &m, &pos);

		auto iaccess = aux::ringproject(pos, m);
		std::printf("i: %d, iaccess: %d\n", pos, iaccess);
	}
}
