#include <cstdio>
#include <algorithm>

template <typename T>
void printarr(T* ar, std::size_t len) {
	for (std::size_t i=0; i<len; ++i) {
		std::printf("%d	\n",ar[i]);
	}
	std::printf("\n");
}

int main () {

	int ar[6] = {1,3,2,6,9,2};
	
	printarr(ar, 6);

	std::sort(ar, ar+6);

	printarr(ar, 6);

	// remove duplicates
	int index = 0;
	for (int m=0; m<6; ++m) {
		if(ar[index] == ar[m]) {
			continue;
		}
		index++;
		ar[index] = ar[m];
	}
	
	int len = index+1;
	std::printf("new len: %d\n",len);

	printarr(ar, len);

}
