#include "pch.h"
#include "Image.h"

using namespace gpu_enhance;

void Image::Reset()
{
	w = 0;
	h = 0;
	size = 0;

	if (rgb != nullptr)
	{
		delete[] rgb;
		rgb = nullptr;
	}
}

Image::Image() :
	w(0),
	h(0),
	rgb(nullptr)
{

}

Image::Image(const CImg<ImgDatatype>* p_cimg) :
	w(0),
	h(0),
	rgb(nullptr)
{
	SetImage(p_cimg);
}

Image::~Image()
{
	Reset();
}

void Image::SetImage(const CImg<ImgDatatype>* p_cimg)
{
	if (p_cimg == nullptr)
		return;

	Reset();

	w = p_cimg->width();
	h = p_cimg->height();

	size = w * h;
	rgb = new RGB[size];

	// Fill
	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			rgb[r * w + c].r = (*p_cimg)(c, r, 0, 0);
			rgb[r * w + c].g = (*p_cimg)(c, r, 0, 1);
			rgb[r * w + c].b = (*p_cimg)(c, r, 0, 2);
			//rgb->a = (*p_cimg)(c, r, 0, 3);
		}
	}
}

void Image::SetRGB(const RGB* p_rgb, const int width, const int height)
{
	if (p_rgb == nullptr)
		return;

	Reset();

	w = width;
	h = height;

	size = w * h;
	rgb = new RGB[size];

	// Fill
	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			rgb[r * w + c].r = p_rgb[r * w + c].r;
			rgb[r * w + c].g = p_rgb[r * w + c].g;
			rgb[r * w + c].b = p_rgb[r * w + c].b;
			//rgb->a = (*p_cimg)(c, r, 0, 3);
		}
	}
}

void Image::CopyToImage(CImg<ImgDatatype>* p_cimg)
{
	if (p_cimg == nullptr || rgb == nullptr)
		return;
	//std::cout << "WORS" << std::endl;
	for (int r = 0; r < h; ++r)
	{
		for (int c = 0; c < w; ++c)
		{
			(*p_cimg)(c, r, 0, 0) = (*this)(c, r).r;
			(*p_cimg)(c, r, 0, 1) = (*this)(c, r).g;
			(*p_cimg)(c, r, 0, 2) = (*this)(c, r).b;
		}
	}
}

const int& Image::GetWidth() const
{
	return w;
}

const int& Image::GetHeight() const
{
	return h;
}

const int& Image::GetSize() const
{
	return size;
}

Image::RGB* Image::GetRGB()
{
	return rgb;
}

const Image::RGB& Image::operator[](const unsigned int& i) const
{
	const int index = (i < size) ? i : size - 1;
	return rgb[i];
}

const Image::RGB& Image::operator()(const unsigned int& x, const unsigned int& y) const
{
	return (*this)[(y * w) + x];
}
