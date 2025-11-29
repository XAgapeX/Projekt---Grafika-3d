#include "negative.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
return; \
}

__global__ void negativeKernel(unsigned char *input,int pitch,unsigned char *output,int width,int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * pitch + x * 3;

    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    output[idx] = 255 - r;
    output[idx + 1] = 255 - g;
    output[idx + 2] = 255 - b;
}

void applyNegativeCUDA(unsigned char *input,unsigned char *output,int width,int height,int pitch) {
    size_t bufferSize = pitch * height;

    unsigned char *d_input, *d_output;
    cudaError_t err;

    err = cudaMalloc(&d_input, bufferSize);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_output, bufferSize);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_input, input, bufferSize, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    negativeKernel<<<blocks, threads>>>(d_input, pitch, d_output, width, height);

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    err = cudaMemcpy(output, d_output, bufferSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    cudaFree(d_input);
    cudaFree(d_output);
}
