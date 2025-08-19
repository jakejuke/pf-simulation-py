#include <cstdio>
#include <src/aux/copyfile.hpp>

int main() {
	
	char srcfilename[32] = "./ops_output.out";
	
	char destfilename[32] = "./ops_output_copy.out";
	
	auto retval = aux::copy_file(srcfilename, destfilename);
	
	return retval;
}
