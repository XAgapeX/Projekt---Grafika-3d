#include "contrast.cuh"
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
return; \
}

__global__ void contrastKernel(unsigned char* input, int inputPitch,
                               unsigned char* output, int width, int height, float factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * inputPitch + x * 3;

    int r = (int)(((input[idx]     - 128) * factor) + 128);
    int g = (int)(((input[idx + 1] - 128) * factor) + 128);
    int b = (int)(((input[idx + 2] - 128) * factor) + 128);

    r = max(0, min(255, r));
    g = max(0, min(255, g));
    b = max(0, min(255, b));

    output[idx]     = (unsigned char)r;
    output[idx + 1] = (unsigned char)g;
    output[idx + 2] = (unsigned char)b;
}

void applyContrastCUDA(unsigned char* input, unsigned char* output,
                       int width, int height, int inputPitch, float factor)
{
    unsigned char *d_input, *d_output;
    size_t size = inputPitch * height;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    contrastKernel<<<blocks, threads>>>(d_input, inputPitch, d_output,
                                        width, height, factor);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
