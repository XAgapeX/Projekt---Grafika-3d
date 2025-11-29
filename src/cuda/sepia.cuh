#pragma once
#include <cuda_runtime.h>

void applySepiaCUDA(unsigned char* input, unsigned char* output,
                    int width, int height, int inputPitch);
