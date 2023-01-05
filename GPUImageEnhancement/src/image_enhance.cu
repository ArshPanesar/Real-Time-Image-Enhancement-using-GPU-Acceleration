#include "image_enhance.cuh"

using namespace gpu_enhance;

long long CPU_SetImageBrightness(gpu_enhance::Image::RGB* dst_rgb, int size, int brightness)
{
    Timer t;
    t.Start("Brightness");


    for (int i = 0; i < size; ++i)
    {
        dst_rgb[i].r = CLAMP_VAL(dst_rgb[i].r + brightness, 0, 255);
        dst_rgb[i].g = CLAMP_VAL(dst_rgb[i].g + brightness, 0, 255);
        dst_rgb[i].b = CLAMP_VAL(dst_rgb[i].b + brightness, 0, 255);
    }

    t.Stop();

    return t.GetDurationInMicroseconds();
}

__global__ void SetImageBrightnessKernel(Image::RGB* dst_rgb, int size, int brightness)
{
    // 128 Threads Per Block
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
    {
        dst_rgb[i].r = CLAMP_VAL((int)dst_rgb[i].r + brightness, 0, 255);
        dst_rgb[i].g = CLAMP_VAL((int)dst_rgb[i].g + brightness, 0, 255);
        dst_rgb[i].b = CLAMP_VAL((int)dst_rgb[i].b + brightness, 0, 255);
    }
}

__global__ void ConvertImageToGrayscale(Image::RGB* dst_rgb, int size)
{
    // 128 Threads Per Block
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
    {
        int luminance = (int)((float)dst_rgb[i].r * 0.2126f) + (int)((float)dst_rgb[i].g * 0.7152f) + (int)((float)dst_rgb[i].b * 0.0722f);
        dst_rgb[i].r = luminance;
        dst_rgb[i].g = luminance;
        dst_rgb[i].b = luminance;
    }
}

__global__ void GenerateGrayLevelHistogramKernel(gpu_enhance::ImgDatatype* histogram, gpu_enhance::Image::RGB* rgb, int size)
{
    // 128 Threads Per Block
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Initialize Histogram Shared By Block
    __shared__ ImgDatatype shared_histogram[HISTOGRAM_SIZE_8_BITS];
    if (HISTOGRAM_SIZE_8_BITS > BLOCK_SIZE)
    {
        for (int i = tx; i < HISTOGRAM_SIZE_8_BITS; i += BLOCK_SIZE)
        {
            if (i < HISTOGRAM_SIZE_8_BITS)
                shared_histogram[i] = 0;
        }
    }
    else
    {
        if (tx < HISTOGRAM_SIZE_8_BITS)
        {
            shared_histogram[tx] = 0;
        }
    }

    __syncthreads();

    // Add Values to Sub Histogram
    int i = tx + bx * blockDim.x;
    if (i < size)
    {
        ImgDatatype luminance = (ImgDatatype)((float)(rgb[i].r + rgb[i].g + rgb[i].b) / 3.0f);
        atomicAdd(&shared_histogram[luminance], 1);
    }

    __syncthreads();

    // Add to Global Histogram
    if (HISTOGRAM_SIZE_8_BITS > BLOCK_SIZE)
    {
        for (int j = tx; j < HISTOGRAM_SIZE_8_BITS; j += BLOCK_SIZE)
        {
            if (j < HISTOGRAM_SIZE_8_BITS)
            {
                atomicAdd(&histogram[j], shared_histogram[j]);
            }
        }
    }
    else
    {
        if (tx < HISTOGRAM_SIZE_8_BITS)
        {
            atomicAdd(&histogram[tx], shared_histogram[tx]);
        }
    }
}

__global__ void SetGrayLevelContrastUsingStreching(gpu_enhance::ImgDatatype lowest, gpu_enhance::ImgDatatype peak, gpu_enhance::Image::RGB* rgb, int size, int contrast)
{
    // 32 Threads Per Block
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int i = tx + bx * blockDim.x;

    ImgDatatype offset = (ImgDatatype)((255.0f) / (float)(peak - lowest));
    
    if (i < size)
    {
        ImgDatatype avg = (ImgDatatype)((float)(rgb[i].r + rgb[i].g + rgb[i].b) / 3.0f);
        ImgDatatype luminance = ((avg - lowest) * offset);

        rgb[i].r = CLAMP_VAL(luminance, 0, 255);
        rgb[i].g = CLAMP_VAL(luminance, 0, 255);
        rgb[i].b = CLAMP_VAL(luminance, 0, 255);
    }
}

void CalculateScaledCumHistogram(gpu_enhance::ImgDatatype* histogram, gpu_enhance::ImgDatatype* scaled_cum_histogram, int image_size)
{
    scaled_cum_histogram[0] = histogram[0];
    for (int i = 1; i < 256; i++)
    {
        scaled_cum_histogram[i] = histogram[i] + scaled_cum_histogram[i - 1];
    }

    float alpha = 255.0f / image_size;
    for (int i = 1; i < 256; i++)
    {
        scaled_cum_histogram[i] = (ImgDatatype)((float)scaled_cum_histogram[i] * alpha);
    }
}

__global__ void SetContrastWithHistogramEqualization(gpu_enhance::ImgDatatype* scaled_cum_histogram, Image::RGB* rgb, int image_size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < image_size)
    {
        ImgDatatype avg = (ImgDatatype)((float)(rgb[i].r + rgb[i].g + rgb[i].b) / 3.0f);
        ImgDatatype val = scaled_cum_histogram[avg];
        rgb[i].r = rgb[i].g = rgb[i].b = CLAMP_VAL(val, 0, 255);
    }
}

__global__ void GenerateImageHistogramKernel(gpu_enhance::ImgDatatype* red_histogram, gpu_enhance::ImgDatatype* green_histogram, gpu_enhance::ImgDatatype* blue_histogram, gpu_enhance::Image::RGB* rgb, int size)
{
    // 128 Threads Per Block
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Initialize Histogram Shared By Block
    __shared__ ImgDatatype shared_red_histogram[HISTOGRAM_SIZE_8_BITS];
    __shared__ ImgDatatype shared_green_histogram[HISTOGRAM_SIZE_8_BITS];
    __shared__ ImgDatatype shared_blue_histogram[HISTOGRAM_SIZE_8_BITS];
    if (HISTOGRAM_SIZE_8_BITS > BLOCK_SIZE)
    {
        for (int i = tx; i < HISTOGRAM_SIZE_8_BITS; i += BLOCK_SIZE)
        {
            if (i < HISTOGRAM_SIZE_8_BITS)
            {
                shared_red_histogram[i] = 0;
                shared_green_histogram[i] = 0;
                shared_blue_histogram[i] = 0;
            }
        }
    }
    else
    {
        if (tx < HISTOGRAM_SIZE_8_BITS)
        {
            shared_red_histogram[tx] = 0;
            shared_green_histogram[tx] = 0;
            shared_blue_histogram[tx] = 0;
        }
    }

    __syncthreads();

    // Add Values to Sub Histogram
    int i = tx + bx * blockDim.x;
    if (i < size)
    {
        atomicAdd(&shared_red_histogram[rgb[i].r], 1);
        atomicAdd(&shared_green_histogram[rgb[i].g], 1);
        atomicAdd(&shared_blue_histogram[rgb[i].b], 1);
    }
    
    __syncthreads();
    
    // Add to Global Histogram
    if (HISTOGRAM_SIZE_8_BITS > BLOCK_SIZE)
    {
        for (int j = tx; j < HISTOGRAM_SIZE_8_BITS; j += BLOCK_SIZE)
        {
            if (j < HISTOGRAM_SIZE_8_BITS)
            {
                atomicAdd(&red_histogram[j], shared_red_histogram[j]);
                atomicAdd(&green_histogram[j], shared_green_histogram[j]);
                atomicAdd(&blue_histogram[j], shared_blue_histogram[j]);
            }
        }
    }
    else
    {
        if (tx < HISTOGRAM_SIZE_8_BITS)
        {
            atomicAdd(&red_histogram[tx], shared_red_histogram[tx]);
            atomicAdd(&green_histogram[tx], shared_green_histogram[tx]);
            atomicAdd(&blue_histogram[tx], shared_blue_histogram[tx]);
        }
    }
}

__global__ void SetImageContrastUsingStreching(gpu_enhance::ImgDatatype red_lowest, gpu_enhance::ImgDatatype red_peak, gpu_enhance::ImgDatatype green_lowest, gpu_enhance::ImgDatatype green_peak, gpu_enhance::ImgDatatype blue_lowest, gpu_enhance::ImgDatatype blue_peak, gpu_enhance::Image::RGB* rgb, int size, int contrast)
{
    // 32 Threads Per Block
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int i = tx + bx * blockDim.x;
    
    ImgDatatype red_offset = (ImgDatatype)((255.0f) / (float)(red_peak - red_lowest));
    ImgDatatype green_offset = (ImgDatatype)((255.0f) / (float)(green_peak - green_lowest));
    ImgDatatype blue_offset = (ImgDatatype)((255.0f) / (float)(blue_peak - blue_lowest));

    if (i < size)
    {
        rgb[i].r = ((rgb[i].r - red_lowest) * red_offset);
        rgb[i].g = ((rgb[i].g - green_lowest) * green_offset);
        rgb[i].b = ((rgb[i].b - blue_lowest) * blue_offset);

        rgb[i].r = CLAMP_VAL(rgb[i].r, 0, 255);
        rgb[i].g = CLAMP_VAL(rgb[i].g, 0, 255);
        rgb[i].b = CLAMP_VAL(rgb[i].b, 0, 255);
    }
}


void SetImageBrightness(gpu_enhance::Image::RGB* dst_rgb, int size, int brightness)
{
    // 32 Threads Per Warp
    // 48 Warps Per SM
    // 30 SMs
    int threadsPerBlock = BLOCK_SIZE; // 128 Best Performance
    int numOfBlocks = (int)std::ceilf((float)size / (float)threadsPerBlock);
    //numOfBlocks = 720; // NO MAJOR IMPACT -> 2% Increase in Occupancy

    //SetImageBrightnessKernel << <numOfBlocks, threadsPerBlock >> > (dst_rgb, size, brightness);
    ConvertImageToGrayscale << <numOfBlocks, threadsPerBlock >> > (dst_rgb, size);
}

void SetImageContrast(gpu_enhance::Image::RGB* dst_rgb, int size, int contrast)
{
    ImgDatatype* host_red_histogram_ptr = new ImgDatatype[HISTOGRAM_SIZE_8_BITS];
    ImgDatatype* host_green_histogram_ptr = new ImgDatatype[HISTOGRAM_SIZE_8_BITS];
    ImgDatatype* host_blue_histogram_ptr = new ImgDatatype[HISTOGRAM_SIZE_8_BITS];
    for (int i = 0; i < HISTOGRAM_SIZE_8_BITS; ++i)
        host_red_histogram_ptr[i] = host_green_histogram_ptr[i] = host_blue_histogram_ptr[i] = 0;
    
    ImgDatatype* dev_red_histogram_ptr = nullptr, *dev_green_histogram_ptr = nullptr, *dev_blue_histogram_ptr = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_red_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS);
    gpue_CUDA_AllocateOnDevice((void**)&dev_green_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS);
    gpue_CUDA_AllocateOnDevice((void**)&dev_blue_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS);
    
    int threadsPerBlock = BLOCK_SIZE;
    int numOfBlocks = (int)std::ceilf((float)size / (float)threadsPerBlock);

    GenerateImageHistogramKernel <<< numOfBlocks, threadsPerBlock >>> (dev_red_histogram_ptr, dev_green_histogram_ptr, dev_blue_histogram_ptr, dst_rgb, size);

    gpue_CUDA_Memcpy(host_red_histogram_ptr, dev_red_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS, cudaMemcpyDeviceToHost);
    gpue_CUDA_Memcpy(host_green_histogram_ptr, dev_green_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS, cudaMemcpyDeviceToHost);
    gpue_CUDA_Memcpy(host_blue_histogram_ptr, dev_blue_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS, cudaMemcpyDeviceToHost);

    // Peak Values in Histogram
    ImgDatatype red_lowest = 255, red_peak = 0;
    ImgDatatype green_lowest = 255, green_peak = 0;
    ImgDatatype blue_lowest = 255, blue_peak = 0;
    for (int i = 0; i < HISTOGRAM_SIZE_8_BITS; ++i)
    {
        // RED
        if (host_red_histogram_ptr[i] > 0 && host_red_histogram_ptr[i] < red_lowest)
            red_lowest = host_red_histogram_ptr[i];
        else if (host_red_histogram_ptr[i] < 255 && host_red_histogram_ptr[i] > red_peak)
            red_peak = host_red_histogram_ptr[i];

        // GREEN
        if (host_green_histogram_ptr[i] > 0 && host_green_histogram_ptr[i] < green_lowest)
            green_lowest = host_green_histogram_ptr[i];
        else if (host_green_histogram_ptr[i] < 255 && host_green_histogram_ptr[i] > green_peak)
            green_peak = host_green_histogram_ptr[i];

        // BLUE
        if (host_blue_histogram_ptr[i] > 0 && host_blue_histogram_ptr[i] < blue_lowest)
            blue_lowest = host_blue_histogram_ptr[i];
        else if (host_blue_histogram_ptr[i] < 255 && host_blue_histogram_ptr[i] > blue_peak)
            blue_peak = host_blue_histogram_ptr[i];
    }

    std::cout << "red: " << red_lowest << ", " << red_peak << "\n";
    std::cout << "green: " << green_lowest << ", " << green_peak << "\n";
    std::cout << "blue: " << blue_lowest << ", " << blue_peak << "\n";
    SetImageContrastUsingStreching << <numOfBlocks, threadsPerBlock >> > (red_lowest, red_peak, green_lowest, green_peak, blue_lowest, blue_peak, dst_rgb, size, 0);
}

void SetGrayLevelImageContrast(gpu_enhance::Image::RGB* dst_rgb, int w, int h)
{
    int size = w * h;

    ImgDatatype* host_histogram_ptr = new ImgDatatype[HISTOGRAM_SIZE_8_BITS];
    for (int i = 0; i < HISTOGRAM_SIZE_8_BITS; ++i)
        host_histogram_ptr[i] = 0;

    ImgDatatype* dev_histogram_ptr = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS);
    
    int threadsPerBlock = BLOCK_SIZE;
    int numOfBlocks = (int)std::ceilf((float)size / (float)threadsPerBlock);
    
    GenerateGrayLevelHistogramKernel << < numOfBlocks, threadsPerBlock >> > (dev_histogram_ptr, dst_rgb, size);
    gpue_CUDA_Memcpy(host_histogram_ptr, dev_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS, cudaMemcpyDeviceToHost);

    ImgDatatype* host_cum_histogram_ptr = new ImgDatatype[HISTOGRAM_SIZE_8_BITS];    
    CalculateScaledCumHistogram(host_histogram_ptr, host_cum_histogram_ptr, size);
    ImgDatatype* dev_cum_histogram_ptr = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_cum_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS);
    gpue_CUDA_Memcpy(dev_cum_histogram_ptr, host_cum_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_8_BITS, cudaMemcpyHostToDevice);

    //SetGrayLevelContrastUsingStreching << < numOfBlocks, threadsPerBlock >> > (lowest, peak, dst_rgb, size, 100);
    SetContrastWithHistogramEqualization << < numOfBlocks, threadsPerBlock >> > (dev_cum_histogram_ptr, dst_rgb, size);
 
    gpue_CUDA_FreeFromDevice(dev_histogram_ptr);
    gpue_CUDA_FreeFromDevice(dev_cum_histogram_ptr);
}