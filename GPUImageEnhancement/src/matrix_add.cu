
#include "matrix_add.cuh"

cudaError_t addWithCuda(int* c, const int* a, const int* b, int N);

__global__ void MatrixAddKernel(int* sum, int* a, int* b, int N)
{
    int i = threadIdx.x;
    sum[i] = a[i] + b[i];
}

void printMatrix(int* m, int N, char* Title)
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

int runMatrixAdd(const int N)
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

    addWithCuda(c, a, b, N);

    //printMatrix(c, N, "Sum Matrix:");

	free(a);
	free(b);
	free(c);
	a = b = c = NULL;
	return 0;
}


cudaError_t addWithCuda(int* c, const int* a, const int* b, int N)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int size = N * N;
    int Error = 0;
    cudaError_t cudaStatus;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    //Timer t1("CUDA Malloc");
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_c);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }
    //t1.Stop();

    // Copy input vectors from host memory to GPU buffers.
    //Timer t2("CUDA Memcpy");
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }
    //t2.Stop();

    // Launch a kernel on the GPU.
    //dim3 threadsPerBlock(N * N);
    //dim3 numBlocks(1, N / threadsPerBlock.y);
    int threadsPerBlock = size;
    //Timer t3("MatrixAddKernel Call");
    MatrixAddKernel<<< 1, threadsPerBlock >>>(dev_c, dev_a, dev_b, N);
    //t3.Stop();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_c);
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}