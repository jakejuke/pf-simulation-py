#include <cstdio>
#include <cstddef>

int main() {

	float value = 42;
	char buf[64];

	std::sprintf(buf, "teststring %f\n", value);

	std::puts(buf);
}
