#include "grayscale.cuh"

__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = (y * width + x) * 3;
        unsigned char r = input[index];
        unsigned char g = input[index + 1];
        unsigned char b = input[index + 2];
        output[y * width + x] = static_cast<unsigned char>((r + g + b) / 3);
    }
}

void applyGrayscaleCUDA(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char *d_input, *d_output;
    size_t imgSize = width * height * 3;
    size_t graySize = width * height;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, graySize);

    cudaMemcpy(d_input, input, imgSize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    grayscaleKernel<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, graySize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
