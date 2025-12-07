#include "cartoon.cuh"
#include <iostream>
#include <cmath>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    return; \
}

__global__ void cartoonKernel(unsigned char* input, int inputPitch,
                              unsigned char* output, int width, int height,
                              int colorLevels, float edgeThreshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * inputPitch + x * 3;


    int r = input[idx];
    int g = input[idx + 1];
    int b = input[idx + 2];

    float lum = 0.299f * r + 0.587f * g + 0.114f * b;
    float factor = 1.2f;
    r = max(0, min(255, (int)((r - lum) * factor + lum)));
    g = max(0, min(255, (int)((g - lum) * factor + lum)));
    b = max(0, min(255, (int)((b - lum) * factor + lum)));

    input[idx]     = (unsigned char)r;
    input[idx + 1] = (unsigned char)g;
    input[idx + 2] = (unsigned char)b;

    for (int c = 0; c < 3; c++) {
        int value = input[idx + c];
        int newValue = (value * max(4, colorLevels)) / 256 * (256 / max(4, colorLevels));
        output[idx + c] = (unsigned char)(max(0, min(255, newValue)));
    }


    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int idx_left  = y * inputPitch + (x - 1) * 3;
        int idx_right = y * inputPitch + (x + 1) * 3;
        int idx_up    = (y - 1) * inputPitch + x * 3;
        int idx_down  = (y + 1) * inputPitch + x * 3;

        int dx = 0, dy = 0;
        for (int c = 0; c < 3; c++) {
            dx += abs(input[idx_right + c] - input[idx_left + c]);
            dy += abs(input[idx_down + c] - input[idx_up + c]);
        }

        float magnitude = sqrtf((float)(dx * dx + dy * dy));
        float dynamicThreshold = edgeThreshold * 255.0f / (lum + 1.0f);
        if (magnitude > dynamicThreshold) {
            output[idx] = output[idx + 1] = output[idx + 2] = 0;
        }
    }
}

void applyCartoonCUDA(unsigned char* input, unsigned char* output,
                      int width, int height, int inputPitch,
                      int colorLevels, float edgeThreshold)
{
    unsigned char *d_input, *d_output;
    size_t size = inputPitch * height;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cartoonKernel<<<blocks, threads>>>(d_input, inputPitch, d_output,
                                       width, height, colorLevels, edgeThreshold);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
