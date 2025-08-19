#include <cstdio>
#include <src/sim/findpos.hpp>
#include <cstddef>

int main() {

	constexpr std::size_t len = 10;
	std::size_t now = 7;
	unsigned short buf[len] = {0,1,2,3,4,5,6,0,0,0};

	unsigned short idtosearch = 7;
	auto pos = sim::findpos(now, buf, idtosearch,2);

	std::printf("id %d at position %d\n", idtosearch, pos);
}
