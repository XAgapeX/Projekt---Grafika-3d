#include "gaussianblur.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

typedef unsigned char uchar;


#define KERNEL_SIZE 21
#define SIGMA 7.0f
#define RADIUS (KERNEL_SIZE / 2)

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

static float h_kernel[KERNEL_SIZE * KERNEL_SIZE];

void generateGaussianKernel() {
    float sum = 0.0f;
    int idx = 0;

    for (int y = -RADIUS; y <= RADIUS; ++y) {
        for (int x = -RADIUS; x <= RADIUS; ++x) {
            float value = expf(-(x*x + y*y) / (2 * SIGMA * SIGMA));
            h_kernel[idx++] = value;
            sum += value;
        }
    }

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i)
        h_kernel[i] /= sum;

    cudaMemcpyToSymbol(d_kernel, h_kernel,
                       KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
}


__global__
void gaussianBlurKernel(const uchar* input, uchar* output,
                        int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float r = 0, g = 0, b = 0;
    int idx = 0;

    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        int py = min(max(y + ky, 0), height - 1);

        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            int px = min(max(x + kx, 0), width - 1);

            const uchar* pixel = input + py * pitch + px * 3;

            float w = d_kernel[idx++];

            r += pixel[0] * w;
            g += pixel[1] * w;
            b += pixel[2] * w;
        }
    }

    uchar* out = output + y * pitch + x * 3;
    out[0] = (uchar)r;
    out[1] = (uchar)g;
    out[2] = (uchar)b;
}

void applyGaussianBlurCUDA(const uchar* input, uchar* output,
                           int width, int height, int pitch)
{
    static bool initialized = false;
    if (!initialized) {
        generateGaussianKernel();
        initialized = true;
    }

    uchar *d_input, *d_output;
    size_t totalSize = pitch * height;

    cudaMalloc(&d_input, totalSize);
    cudaMalloc(&d_output, totalSize);

    cudaMemcpy(d_input, input, totalSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, width, height, pitch);

    cudaMemcpy(output, d_output, totalSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
