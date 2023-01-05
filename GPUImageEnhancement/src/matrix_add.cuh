#include "pch.h"
#include "Timer.h"

__global__ void MatrixAddKernel(int* sum, int* a, int* b, int N);

int runMatrixAdd(int N);