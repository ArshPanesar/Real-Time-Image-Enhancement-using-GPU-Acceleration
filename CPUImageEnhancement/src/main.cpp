#include "matrix_add.h"
#include <math.h>

int main()
{
	printf("Running on CPU:\n");
	
	const int NUM_OF_TESTS = 10;
	int N = 2;
	for (int i = 0; i < NUM_OF_TESTS; ++i) 
	{
		std::cout << "N : " << N << std::endl;
		
		Timer t;
		runMatrixAddWithCPU(N);
		t.Stop();
		
		N += 2;
	}


	return 0;
}