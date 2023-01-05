#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cooperative_groups.h"

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <iostream>
#include <chrono>

#define cimg_use_tiff
#define cimg_use_png
#define cimg_use_jpeg
#include "CImg.h"

#define MIN_VAL(a, b) ((a) < (b)) ? (a) : (b)
#define MAX_VAL(a, b) ((a) > (b)) ? (a) : (b)

#define CLAMP_VAL(val, min, max) ( MIN_VAL((val), (min)) == (val) ) ? (min) : ( (MAX_VAL((val), (max)) == (val)) ? (max) : (val) )
#define ROUND_VAL(val) \
{ \
	int ival = (int)val; \
	double offset = val - (double)ival; \
	val = (offset < 0.5f) ? (double)(ival) : (double)(ival + 1); \
}