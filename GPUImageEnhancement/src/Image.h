#pragma once
#include "pch.h"
#include "CImg.h"
#include "Timer.h"

using namespace cimg_library;

namespace gpu_enhance
{
	typedef unsigned int ImgDatatype;

	// Image Representation
	class Image
	{
	public:
		// RGB Represenation
		struct RGB
		{
			ImgDatatype r;
			ImgDatatype g;
			ImgDatatype b;
			ImgDatatype a;
		};

	protected:


		RGB* rgb;
		int w;
		int h;
		int size;

	protected:
		void Reset();

	public:
		Image();
		Image(const CImg<ImgDatatype>* p_cimg);
		~Image();

		void SetImage(const CImg<ImgDatatype>* p_cimg);
		void SetRGB(const RGB* p_rgb, const int width, const int height);
		void CopyToImage(CImg<ImgDatatype>* p_cimg);

		const int& GetWidth() const;
		const int& GetHeight() const;
		const int& GetSize() const; // Returns Width * Height
		RGB* GetRGB();


		const RGB& operator[](const unsigned int& i) const;
		const RGB& operator()(const unsigned int& x, const unsigned int& y) const;
	};
}