#include "pch.h"
#include "Application.h"

using namespace gpu_enhance;

void gpu_enhance::Application::CopyToDevice()
{
	if (dev_img_ptr == nullptr)
	{
		gpue_CUDA_AllocateOnDevice((void**)&dev_img_ptr, sizeof(Image::RGB) * app_img.GetSize());
	}

	gpue_CUDA_Memcpy(dev_img_ptr, app_img.GetRGB(), sizeof(Image::RGB)* app_img.GetSize(), cudaMemcpyHostToDevice);
}

void gpu_enhance::Application::CopyToHost()
{
	if (host_img_ptr == nullptr)
	{
		gpue_CUDA_AllocateOnHost((void**)&host_img_ptr, sizeof(Image::RGB) * app_img.GetSize(), cudaHostAllocPortable);
	}

	gpue_CUDA_Memcpy(host_img_ptr, dev_img_ptr, sizeof(Image::RGB) * app_img.GetSize(), cudaMemcpyDeviceToHost);
}

gpu_enhance::Application::Application() :
	cimg(),
	app_img(),
	img_path(),
	host_img_ptr(nullptr),
	dev_img_ptr(nullptr),
	IsKernelRunning(false)
{
}

Application::~Application()
{
	if (host_img_ptr != nullptr)
		cudaFreeHost(host_img_ptr);
	if (dev_img_ptr != nullptr)
		cudaFree(dev_img_ptr);

	host_img_ptr = dev_img_ptr = nullptr;
}

void gpu_enhance::Application::LoadImg(const std::string& ImagePath)
{
	img_path = ImagePath;

	cimg.load(img_path.c_str());
	cimg.blur(0.0);

	app_img.SetImage(&cimg);

	if (host_img_ptr != nullptr)
		cudaFree(host_img_ptr);
	if (dev_img_ptr != nullptr)
		cudaFree(dev_img_ptr);

	host_img_ptr = dev_img_ptr = nullptr;

	// Send Image to GPU
	CopyToDevice();
	CopyToHost();
}

void gpu_enhance::Application::SaveImg(const std::string& NewName)
{
	cimg.save(NewName.c_str());
}

const CImg<ImgDatatype>& gpu_enhance::Application::GetCImg() const
{
	return cimg;
}

const Image& gpu_enhance::Application::GetAppImg() const
{
	return app_img;
}
