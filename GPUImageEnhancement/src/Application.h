#pragma once
#include "pch.h"
#include "Image.h"
#include "Core.cuh"

namespace gpu_enhance
{
	class Application
	{
	protected:
		CImg<ImgDatatype> cimg;
		Image app_img;

		std::string img_path;

		// Pointers
		Image::RGB* host_img_ptr;
		Image::RGB* dev_img_ptr;
		
		// Copies Source Image from Host to Device (GPU)
		void CopyToDevice();

		// Copies Image from GPU to Device
		void CopyToHost();

		bool IsKernelRunning;
	public:
		Application();
		~Application();

		void LoadImg(const std::string& ImagePath);
		void SaveImg(const std::string& NewName);

		virtual void Init() {};
		virtual void Update() = 0;
		virtual void Destroy() {};

		const CImg<ImgDatatype>& GetCImg() const;
		const Image& GetAppImg() const;
	};
}