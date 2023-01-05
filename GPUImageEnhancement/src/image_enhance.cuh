#pragma once
#include "pch.h"
#include "Image.h"
#include "Core.cuh"


// CPU
long long CPU_SetImageBrightness(gpu_enhance::Image::RGB* dst_rgb, int size, int brightness);
__global__ void SetImageBrightnessKernel(gpu_enhance::Image::RGB* dst_rgb, int size, int brightness);
__global__ void ConvertImageToGrayscale(gpu_enhance::Image::RGB* dst_rgb, int size);

__global__ void GenerateGrayLevelHistogramKernel(gpu_enhance::ImgDatatype* histogram, gpu_enhance::Image::RGB* rgb, int size);
__global__ void SetGrayLevelContrastUsingStreching(gpu_enhance::ImgDatatype lowest, gpu_enhance::ImgDatatype peak, gpu_enhance::Image::RGB* rgb, int size, int contrast);

__global__ void GenerateImageHistogramKernel(gpu_enhance::ImgDatatype* red_histogram, gpu_enhance::ImgDatatype* green_histogram, gpu_enhance::ImgDatatype* blue_histogram, gpu_enhance::Image::RGB* rgb, int size);
__global__ void SetImageContrastUsingStreching(gpu_enhance::ImgDatatype red_lowest, gpu_enhance::ImgDatatype red_peak, gpu_enhance::ImgDatatype green_lowest, gpu_enhance::ImgDatatype green_peak, gpu_enhance::ImgDatatype blue_lowest, gpu_enhance::ImgDatatype blue_peak, gpu_enhance::Image::RGB* rgb, int size, int contrast);


void SetImageBrightness(gpu_enhance::Image::RGB* dst_rgb, int size, int brightness);
void SetImageContrast(gpu_enhance::Image::RGB* dst_rgb, int size, int contrast);
void SetGrayLevelImageContrast(gpu_enhance::Image::RGB* dst_rgb, int w, int h);
