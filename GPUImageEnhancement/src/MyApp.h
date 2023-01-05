#pragma once
#include "Application.h"
#include "image_enhance.cuh"
#include "joint_histogram_equalization.cuh"

class MyApp : public gpu_enhance::Application
{
public:
	MyApp();
	~MyApp();

	void Update();
	void UpdateWithMemTransfer();

	void RunNormContrastEnhance();
	void RunJointContrastEnhance();
};
