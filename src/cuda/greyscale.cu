#include "grayscale.cuh"
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
return; \
}

__global__ void grayscaleKernel(unsigned char* input, int inputPitch,
                                unsigned char* output, int width, int height)       {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int inputIdx = y * inputPitch + x * 3;
        unsigned char r = input[inputIdx];
        unsigned char g = input[inputIdx + 1];
        unsigned char b = input[inputIdx + 2];
        output[y * width + x] = static_cast<unsigned char>((r + g + b) / 3);
    }
}

void applyGrayscaleCUDA(unsigned char* input, unsigned char* output,
                        int width, int height, int inputPitch)
{
    unsigned char *d_input, *d_output;
    size_t graySize = width * height;
    size_t inputSize = inputPitch * height;

    cudaError_t err;

    err = cudaMalloc(&d_input, inputSize);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_output, graySize);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<blocks, threads>>>(d_input, inputPitch, d_output, width, height);

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    err = cudaMemcpy(output, d_output, graySize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    cudaFree(d_input);
    cudaFree(d_output);
}
