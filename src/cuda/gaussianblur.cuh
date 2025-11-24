#ifndef GAUSSIANBLUR_CUH
#define GAUSSIANBLUR_CUH

#include <cuda_runtime.h>
#include <stdint.h>

typedef unsigned char uchar;

void applyGaussianBlurCUDA(const uchar* input, uchar* output,
                           int width, int height, int pitch);

#endif
