#pragma once
#include <cuda_runtime.h>

void applyGrayscaleCUDA(unsigned char* input,
                        unsigned char* outputY,
                        unsigned char* outputCb,
                        unsigned char* outputCr,
                        int width, int height, int inputPitch);
