#pragma once
#include "cuda_runtime.h"

void applyGrayscaleCUDA(unsigned char* input, unsigned char* output,
                        int width, int height, int inputPitch);
