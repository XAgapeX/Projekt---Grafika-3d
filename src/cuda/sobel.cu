#include "sobel.cuh"
#include <cuda_runtime.h>
#include <math.h>

typedef unsigned char uchar;

// Sobel w poziomie (Gx)
__constant__ int d_sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Sobel w pionie (Gy)
__constant__ int d_sobelY[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};


__device__
inline float luminance(const uchar* p)
{
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
}


__global__
void sobelKernel(const uchar* input, uchar* output,
                 int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float gx = 0.0f;
    float gy = 0.0f;

    int idx = 0;

    // Sobel 3×3
    for (int ky = -1; ky <= 1; ky++)
    {
        int py = min(max(y + ky, 0), height - 1);

        for (int kx = -1; kx <= 1; kx++)
        {
            int px = min(max(x + kx, 0), width - 1);

            const uchar* pixel = input + py * pitch + px * 3;

            float L = luminance(pixel);

            gx += L * d_sobelX[idx];
            gy += L * d_sobelY[idx];

            idx++;
        }
    }

    float mag = sqrtf(gx * gx + gy * gy);

    if (mag > 255) mag = 255;

    uchar edge = (uchar)mag;

    uchar* out = output + y * pitch + x * 3;

    // wynik w 3 kanałach (biały = krawędź)
    out[0] = edge;
    out[1] = edge;
    out[2] = edge;
}


void applySobelCUDA(const uchar* input, uchar* output,
                    int width, int height, int pitch)
{
    uchar *d_input, *d_output;
    size_t size = pitch * height;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    sobelKernel<<<grid, block>>>(d_input, d_output, width, height, pitch);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
