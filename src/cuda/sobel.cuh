#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <cuda_runtime.h>
#include <stdint.h>

typedef unsigned char uchar;

void applySobelCUDA(const uchar* input, uchar* output,
                    int width, int height, int pitch);

#endif
