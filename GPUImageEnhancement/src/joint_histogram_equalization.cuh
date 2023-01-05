#pragma once
#include "pch.h"
#include "Image.h"
#include "Core.cuh"

// Joint Histogram Equalization: Implemented in CUDA
__global__ void AverageGrayscaledImage(gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int width, const int height);
__global__ void BuildGrayscaledJointHistogram(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int size);
void BuildCDFJointHistogram(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::ImgDatatype* cdf, int* min_cdf, int size);
__global__ void EqualizeGrayscaledJointHistogram(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::ImgDatatype* cdf, int* min_cdf, gpu_enhance::Image::RGB* img, const int size_dec);
__global__ void ApplyJointHistgramToGrascaledImage(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int size);

void SetGrayscaledContrastWithJointHistogramEqualization(gpu_enhance::Image::RGB* img, const int width, const int height);
void CPU_SetGrayscaledContrastWithJointHistogramEqualization(gpu_enhance::Image::RGB* img, const int width, const int height);
void DenoiseImageWithAverageFilter(gpu_enhance::Image::RGB* img, const int width, const int height);

long long CPU_AverageImage(gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int width, const int height);
