#include "sepia.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
    << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    return; \
}

__device__ inline unsigned char clampToByte(float x) {
    if (x < 0.f) return 0;
    if (x > 255.f) return 255;
    return static_cast<unsigned char>(x);
}

__global__ void sepiaKernel(unsigned char* input, int inputPitch,
                            unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * inputPitch + x * 3;

        unsigned char r = input[idx + 0];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        float oR = 0.393f*r + 0.769f*g + 0.189f*b;
        float oG = 0.349f*r + 0.686f*g + 0.168f*b;
        float oB = 0.272f*r + 0.534f*g + 0.131f*b;

        int outIdx = y * width * 3 + x * 3;

        output[outIdx + 0] = clampToByte(oR);
        output[outIdx + 1] = clampToByte(oG);
        output[outIdx + 2] = clampToByte(oB);
    }
}

void applySepiaCUDA(unsigned char* input, unsigned char* output,
                    int width, int height, int inputPitch)
{
    unsigned char *d_input, *d_output;

    size_t inputSize = inputPitch * height;
    size_t outputSize = width * height * 3;

    cudaError_t err;

    err = cudaMalloc(&d_input, inputSize);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_output, outputSize);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    sepiaKernel<<<blocks, threads>>>(d_input, inputPitch, d_output, width, height);

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    err = cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    cudaFree(d_input);
    cudaFree(d_output);
}
