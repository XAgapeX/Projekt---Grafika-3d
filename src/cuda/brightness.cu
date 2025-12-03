#include "brightness.cuh"
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
return; \
}

__global__ void brightnessKernel(unsigned char* input, int inputPitch,
                                 unsigned char* output, int width, int height, int brightness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * inputPitch + x * 3;

    int r = input[idx]     + brightness;
    int g = input[idx + 1] + brightness;
    int b = input[idx + 2] + brightness;

    r = max(0, min(255, r));
    g = max(0, min(255, g));
    b = max(0, min(255, b));

    output[idx]     = static_cast<unsigned char>(r);
    output[idx + 1] = static_cast<unsigned char>(g);
    output[idx + 2] = static_cast<unsigned char>(b);
}

void applyBrightnessCUDA(unsigned char* input, unsigned char* output,
                         int width, int height, int inputPitch, int brightness)
{
    unsigned char *d_input, *d_output;
    size_t size = inputPitch * height;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    brightnessKernel<<<blocks, threads>>>(d_input, inputPitch, d_output,
                                          width, height, brightness);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
