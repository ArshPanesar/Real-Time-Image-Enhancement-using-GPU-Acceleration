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

## Profiling Results
These are the profiling results done for Image Brightening and Image Averaging showing that some of these algorithms can be used in Real-Time Applications, when keeping a soft deadline of 1 millisecond. However, more of these algorithms can run in Real-Time, for example, the Joint Histogram Equalization currently uses the CPU to calculate the Cumulative Density Function (CDF) for each bin in the generated histogram, which may be done on the GPU using a Parallel Scan.

> ![image](https://user-images.githubusercontent.com/43693790/210785805-8339fe37-678d-4f5a-b248-8ffa3dc6c9f8.png)
> ![image](https://user-images.githubusercontent.com/43693790/210785850-05721ac6-f640-4c1c-b1b1-f71996bdca8c.png)


