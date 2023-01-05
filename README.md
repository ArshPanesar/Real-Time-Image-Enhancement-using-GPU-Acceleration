# GPUImageEnhancement
This repository contains the following Image Enhancement Algorithms implemented in NVIDIA's CUDA API:
- Image Brightening/Darkening
- Image Contrast Enhancement:
    - Histogram Streching
    - Histogram Equalization
    - Joint Histogram Equalization (Based on “A Novel Joint Histogram Equalization based Image Contrast Enhancement” published by Sanjay Agrawal, Rutuparna Panda, P.K. Mishro and Ajith Abraham)
- Image Averaging
- Image Grayscaling

## About
I made this project for a case study on Image Processing and Accelerating Parallel Computations using the GPU. Some of the results of the various Image Enhancement Algorithms implemented in this project are given below.

## Image Enhancement Results

### Image Brightness
> ![before brightness](GPUImageEnhancement/resources/report/brightness_before.png) ![after brightness](GPUImageEnhancement/resources/report/brightness_after.png)

### Image Averaging (Denoising)
> ![before averaging](GPUImageEnhancement/resources/report/average_before.png) ![after averaging](GPUImageEnhancement/resources/report/average_after.png)

### Image Contrast Enhancement: Histogram Equalization
> ![before contrast histogram equalization](GPUImageEnhancement/resources/report/contrast_equalize_before1.bmp) ![after contrast histogram equalization](GPUImageEnhancement/resources/report/contrast_equalize_after1.png)

### Image Contrast Enhancement: Joint Histogram Equalization
> ![before contrast joint histogram equalization](GPUImageEnhancement/resources/report/jhe_before.bmp) ![after contrast joint histogram equalization](GPUImageEnhancement/resources/report/jhe_after.png)

