#include "joint_histogram_equalization.cuh"

using namespace gpu_enhance;

const int W = 3;
constexpr int K = (int)((float)W / 2.0f);

__global__ void AverageGrayscaledImage(gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int width, const int height)
{
    int size = width * height;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute average image
    if (i < size)
    {
        //int x = i % width;
        //int y = (int)((float)i / (float)width);
        int x = (int)((float)i / (float)width);
        int y = i % width;

        ImgDatatype avg_luminance = 0;
        for (int m = -K; m <= K; ++m)
        {
            for (int n = -K; n <= K; ++n)
            {
                int global_x = x + m;
                int global_y = y + n;
                int global_index = global_y + (global_x * width);

                if (global_index >= 0 && global_index < size)
                    avg_luminance += img[global_index].r;
            }
        }

        avg_luminance = (ImgDatatype)((float)avg_luminance / (float)(W * W));
        avg[i].r = avg[i].g = avg[i].b = avg_luminance;
    }
}

__global__ void BuildGrayscaledJointHistogram(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < size)
    {
        ImgDatatype index = avg[i].r + img[i].r * HISTOGRAM_SIZE_8_BITS;
        //printf("%d\n", img[i].r, avg[i].r, img[i].r != avg[i].r);
        atomicAdd(&joint_histogram[index], 1);
    }
}

void BuildCDFJointHistogram(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::ImgDatatype* cdf, int* min_cdf, int size)
{
    int min_CDF = size + 1;
    ImgDatatype cumulative = 0;
    for (int i = 0; i < HISTOGRAM_SIZE_SQUARED_8_BITS; ++i)
    {
        cumulative += joint_histogram[i];
        min_CDF = (cumulative > 0 && cumulative < min_CDF) ? cumulative : min_CDF;
        cdf[i] = cumulative;
    }

    *min_cdf = min_CDF;  
}

__global__ void EqualizeGrayscaledJointHistogram(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::ImgDatatype* cdf, int* min_cdf, gpu_enhance::Image::RGB* img, const int size_dec)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < HISTOGRAM_SIZE_SQUARED_8_BITS)
    {
        //int x = (int) ((float)i / (float)HISTOGRAM_SIZE_8_BITS);
        //int y = i % HISTOGRAM_SIZE_8_BITS;
        //y + x * HISTOGRAM_SIZE_8_BITS
        double fval = (255.0 * (double)((int)cdf[i] - (int)*min_cdf) / (double)(size_dec));
        //ROUND_VAL(fval);
        ImgDatatype val = (ImgDatatype)fval;
        joint_histogram[i] = val;// CLAMP_VAL(val, 0, 255);
    }
}

__global__ void ApplyJointHistgramToGrascaledImage(gpu_enhance::ImgDatatype* joint_histogram, gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < size)
    {
        ImgDatatype index = avg[i].r + img[i].r * HISTOGRAM_SIZE_8_BITS;
        img[i].r = img[i].g = img[i].b = joint_histogram[index];
    }
}

void SetGrayscaledContrastWithJointHistogramEqualization(gpu_enhance::Image::RGB* img, const int width, const int height)
{
    // Allocating
    int size = width * height;

    // Joint Histogram
    ImgDatatype* dev_joint_histogram_ptr = nullptr, *host_joint_histogram_ptr = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_joint_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS);
    gpue_CUDA_AllocateOnHost((void**)&host_joint_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS);
    //host_joint_histogram_ptr = new ImgDatatype[HISTOGRAM_SIZE_SQUARED_8_BITS];
    cudaMemset(dev_joint_histogram_ptr, 0, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS);

    // CDF
    ImgDatatype* dev_cdf_ptr = nullptr, *host_cdf_ptr = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_cdf_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS);
    gpue_CUDA_AllocateOnHost((void**)&host_cdf_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS);
    cudaMemset(dev_cdf_ptr, 0, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS);

    //host_cdf_ptr = new ImgDatatype[HISTOGRAM_SIZE_SQUARED_8_BITS];

    // Minimum CDF stored on GPU Memory
    int* dev_min_cdf = nullptr, *host_min_cdf = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_min_cdf, sizeof(int));
    gpue_CUDA_AllocateOnHost((void**)&host_min_cdf, sizeof(int));

    // Average Image
    Image::RGB* dev_avg = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_avg, sizeof(Image::RGB) * size);
    cudaMemset(dev_avg, 0, sizeof(Image::RGB) * size);

    // Launch Configuration
    int threadsPerBlock = BLOCK_SIZE;
    int numOfBlocks = (int)std::ceilf((float)size / (float)threadsPerBlock);

    //gpue_CUDA_CheckError(cudaGetLastError(), "After Allocation", "gpue_CUDA_AllocateOnDevice");
    // Running Kernels
    AverageGrayscaledImage<<< numOfBlocks, threadsPerBlock >>>(img, dev_avg, width, height);
    BuildGrayscaledJointHistogram << < numOfBlocks, threadsPerBlock >> > (dev_joint_histogram_ptr, img, dev_avg, size);

    //gpue_CUDA_CheckError(cudaGetLastError(), "Running First 2 Kernels", "Average and BuildHistogram");
    
    //BuildCDFJointHistogram << <numOfBlocks, threadsPerBlock >> > (dev_joint_histogram_ptr, dev_cdf_ptr, dev_min_cdf, size);
    cudaMemcpy(host_joint_histogram_ptr, dev_joint_histogram_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS, cudaMemcpyDeviceToHost);
    
    BuildCDFJointHistogram(host_joint_histogram_ptr, host_cdf_ptr, host_min_cdf, size);
    cudaMemcpy(dev_min_cdf, host_min_cdf, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cdf_ptr, host_cdf_ptr, sizeof(ImgDatatype) * HISTOGRAM_SIZE_SQUARED_8_BITS, cudaMemcpyHostToDevice);

//    gpue_CUDA_CheckError(cudaGetLastError(), "Building CDF", "cudaMemcpy");


    EqualizeGrayscaledJointHistogram << <numOfBlocks, threadsPerBlock >> > (dev_joint_histogram_ptr, dev_cdf_ptr, dev_min_cdf, img, size - 1);
    ApplyJointHistgramToGrascaledImage << <numOfBlocks, threadsPerBlock >> > (dev_joint_histogram_ptr, img, dev_avg, size);

    //gpue_CUDA_CheckError(cudaGetLastError(), "Equalizing and Applying", "kernels");

    // Free resources
    gpue_CUDA_FreeFromDevice(dev_avg);
    gpue_CUDA_FreeFromDevice(dev_min_cdf);
    gpue_CUDA_FreeFromDevice(dev_cdf_ptr);
    gpue_CUDA_FreeFromDevice(dev_joint_histogram_ptr);

    gpue_CUDA_FreeFromHost(host_cdf_ptr);
    gpue_CUDA_FreeFromHost(host_joint_histogram_ptr);
    gpue_CUDA_FreeFromHost(host_min_cdf);
    //gpue_CUDA_CheckError(cudaGetLastError(), "After Freeing", "gpue_CUDA_FreeFromDevice");

    //delete[] host_cdf_ptr;
    //delete[] host_joint_histogram_ptr;

    //std::cout << "HERE\n";
    //cudaDeviceSynchronize();
}





void CPU_SetGrayscaledContrastWithJointHistogramEqualization(gpu_enhance::Image::RGB* img, const int width, const int height)
{
    int size = width * height;
    Image::RGB* avg = new Image::RGB[size];
    int jh[HISTOGRAM_SIZE_8_BITS][HISTOGRAM_SIZE_8_BITS];
    int cdf[HISTOGRAM_SIZE_8_BITS][HISTOGRAM_SIZE_8_BITS];

    for (int i = 0; i < HISTOGRAM_SIZE_8_BITS; ++i)
    {
        for (int j = 0; j < HISTOGRAM_SIZE_8_BITS; ++j)
        {
            jh[i][j] = 0;
            cdf[i][j] = 0;
        }
    }

    // Average Image
    for (int x = 0; x < height; ++x)
    {
        for (int y = 0; y < width; ++y)
        {
            ImgDatatype avg_luminance = 0;
            for (int m = -K; m <= K; ++m)
            {
                for (int n = -K; n <= K; ++n)
                {
                    int global_x = x + m;
                    int global_y = y + n;
                    int global_index = global_y + (global_x * width);

                    if (global_index >= 0 && global_index < size)
                        avg_luminance += img[global_index].r;
                }
            }

            avg_luminance = (ImgDatatype)((float)avg_luminance / (float)(W * W));
            int i = y + x * width;
            avg[i].r = avg[i].g = avg[i].b = avg_luminance;
        }
    }

    // Building JH
    for (int i = 0; i < size; ++i)
    {
        int x = img[i].r;
        int y = avg[i].r;
        jh[x][y] += 1;
    }
    
    // Building CDF
    int min_CDF = size + 1;
    int cumulative = 0;
    for (int i = 0; i < HISTOGRAM_SIZE_8_BITS; i++) 
    {
        for (int j = 0; j < HISTOGRAM_SIZE_8_BITS; j++) 
        {
            int count = jh[i][j];
            cumulative += count;
            if (cumulative > 0 && cumulative < min_CDF)
                min_CDF = cumulative;
            cdf[i][j] = cumulative;
        }
    }

    // Equalizing JH
    ImgDatatype jh_eq[HISTOGRAM_SIZE_8_BITS][HISTOGRAM_SIZE_8_BITS];
    for (int i = 0; i < HISTOGRAM_SIZE_8_BITS; i++) 
    {
        for (int j = 0; j < HISTOGRAM_SIZE_8_BITS; j++) 
        {
            int cur_cdf = cdf[i][j];
            //h_eq_it[j] = cv::saturate_cast<uchar>(255.0 * (cur_cdf - min_CDF) / (total_pixels - 1));
            jh_eq[i][j] = (255.0 * (cur_cdf - min_CDF) / (double)(size - 1));
        }
    }

    // Applying to Image
    for (int i = 0; i < size; ++i)
    {
        int x = img[i].r;
        int y = avg[i].r;

        img[i].r = img[i].g = img[i].b = jh_eq[x][y];
    }

    delete avg;
}





void DenoiseImageWithAverageFilter(gpu_enhance::Image::RGB* img, const int width, const int height)
{
    // Allocating
    int size = width * height;
    
    // Average Image
    Image::RGB* dev_avg = nullptr, * host_avg = nullptr;
    gpue_CUDA_AllocateOnDevice((void**)&dev_avg, sizeof(Image::RGB) * size);

    // Launch Configuration
    int threadsPerBlock = BLOCK_SIZE;
    int numOfBlocks = (int)std::ceilf((float)size / (float)threadsPerBlock);

    // Running Kernels
    AverageGrayscaledImage << < numOfBlocks, threadsPerBlock >> > (img, dev_avg, width, height);
    
    gpue_CUDA_Memcpy(img, dev_avg, sizeof(Image::RGB) * size, cudaMemcpyDeviceToDevice);

    // Free resources
    gpue_CUDA_FreeFromDevice(dev_avg);
}


long long CPU_AverageImage(gpu_enhance::Image::RGB* img, gpu_enhance::Image::RGB* avg, const int width, const int height)
{
    Timer t;
    t.Start("some");

    int size = width * height;
    for (int i = 0; i < size; ++i)
    {
        int x = i % width;
        int y = (int)((float)i / (float)width);

        ImgDatatype avg_luminance = 0;
        for (int m = -K; m <= K; ++m)
        {
            for (int n = -K; n <= K; ++n)
            {
                int global_x = x + m;
                int global_y = y + n;
                int global_index = global_x + (global_y * width);

                if (global_index >= 0 && global_index < size)
                    avg_luminance += img[global_index].r;
            }
        }

        avg_luminance = (ImgDatatype)((float)avg_luminance / (float)(W * W));
        avg[i].r = avg[i].g = avg[i].b = avg_luminance;
    }

    t.Stop();

    return t.GetDurationInMicroseconds();
}
