#pragma once
#include "pch.h"
#include "Image.h"

// 128 Best Performance
#define BLOCK_SIZE 128
// Histogram Size for 8-Bit Image
#define HISTOGRAM_SIZE_8_BITS 256
// Histogram Size Squared
#define HISTOGRAM_SIZE_SQUARED_8_BITS 65536


// CUDA Check if any Errors Occured
// WhoCalled: Which Function called the CUDA Function
// WhatCalled: Which CUDA Function was Called
bool gpue_CUDA_CheckError(cudaError status, const char* WhoCalled, const char* WhatCalled);

// CUDA Memory Allocation
bool gpue_CUDA_AllocateOnHost(void** ptr, size_t size, unsigned int flags = 1U);
bool gpue_CUDA_AllocateOnDevice(void** ptr, size_t size, unsigned int flags = 1U);
bool gpue_CUDA_FreeFromDevice(void* dev_ptr);
bool gpue_CUDA_FreeFromHost(void* host_ptr);

// CUDA Memcpy
bool gpue_CUDA_Memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
