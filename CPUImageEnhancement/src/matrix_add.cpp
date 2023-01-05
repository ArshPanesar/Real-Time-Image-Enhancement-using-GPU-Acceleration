#include "matrix_add.h"

void printMatrix(int* m, int N, const char* Title)
{
	printf("%s\n", Title);
	int Index = 0;
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			Index = (i * N) + j;
			printf("%d  ", m[Index]);
		}
		printf("\n");
	}
}

int runMatrixAddWithCPU(int N)
{
	//printf("Mat Dimensions: %d\n", N);

	// Allocating Matrices on Host
	int* a = (int*)malloc(sizeof(int) * N * N);
	int* b = (int*)malloc(sizeof(int) * N * N);
	int* c = (int*)malloc(sizeof(int) * N * N);

	// Filling Matrices
	int a_offset = 23;
	int b_offset = 67;
	for (int i = 0; i < (N * N); ++i)
	{
		a[i] = i + (i * a_offset);
		b[i] = i + (i * b_offset);
		c[i] = 0;
	}

	//printMatrix(a, N, "Matrix A:");
	//printMatrix(b, N, "Matrix B:");

	addWithCPU(c, a, b, N);
	
	//printMatrix(c, N, "Sum Matrix:");

	free(a);
	free(b);
	free(c);
	a = b = c = NULL;
	return 0;
}

void addWithCPU(int* c, int* a, int* b, int N)
{
	int size = (N * N);
	for (int i = 0; i < size; ++i)
		c[i] = a[i] + b[i];
}