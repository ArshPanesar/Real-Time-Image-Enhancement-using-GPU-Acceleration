#include "Core.cuh"

using namespace gpu_enhance;

bool gpue_CUDA_CheckError(cudaError status, const char* WhoCalled, const char* WhatCalled) 
{
	if (status != cudaSuccess)
	{
		std::cerr << WhoCalled << ": " << WhatCalled << " failed! CudaStatus: " << cudaGetErrorString(status) << std::endl;
		return true;
	}
	return false;
}

bool gpue_CUDA_AllocateOnHost(void** ptr, size_t size, unsigned int flags)
{
	cudaError status = cudaHostAlloc(ptr, size, flags);
	if (gpue_CUDA_CheckError(status, "gpue_CUDA_AllocateOnHost", "cudaHostAlloc"))
	{
		cudaFree(ptr);
		return false;
	}

	return true;
}

bool gpue_CUDA_AllocateOnDevice(void** ptr, size_t size, unsigned int flags)
{
	cudaError status = cudaMallocManaged(ptr, size, flags);
	if (gpue_CUDA_CheckError(status, "gpue_CUDA_AllocateOnDevice", "cudaMallocManaged"))
	{
		cudaFree(ptr);
		return false;
	}

	return true;
}

bool gpue_CUDA_FreeFromDevice(void* dev_ptr)
{
	cudaError status = cudaFree(dev_ptr);
	return !gpue_CUDA_CheckError(status, "gpue_CUDA_FreeFromDevice", "cudaFree");
}

bool gpue_CUDA_FreeFromHost(void* host_ptr)
{
	cudaError status = cudaFreeHost(host_ptr);
	return !gpue_CUDA_CheckError(status, "gpue_CUDA_FreeFromHost", "cudaFreeHost");;
}

bool gpue_CUDA_Memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
	cudaError status = cudaMemcpy(dst, src, count, kind);
	return !gpue_CUDA_CheckError(status, "gpue_CUDA_Memcpy", "cudaMemcpy");
}