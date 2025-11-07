#pragma once

#include "cuda_runtime.h"

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height);
void applyGrayscaleCUDA(unsigned char* input, unsigned char* output, int width, int height);
