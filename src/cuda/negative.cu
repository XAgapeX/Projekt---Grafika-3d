#include "negative.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
return; \
}

/**
 * @brief Kernel CUDA nakładający efekt negatywu na obraz RGB.
 *
 * Dla każdego piksela w obrazie oblicza wartości:
 *   R' = 255 - R
 *   G' = 255 - G
 *   B' = 255 - B
 *
 * @param input  Bufor wejściowy RGB (GPU memory).
 * @param pitch  Pitch (ilość bajtów w wierszu) bufora wejściowego.
 * @param output Bufor wyjściowy RGB (GPU memory).
 * @param width  Szerokość obrazu w pikselach.
 * @param height Wysokość obrazu w pikselach.
 */
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

/**
 * @brief Funkcja nakładająca filtr negatywu na obraz RGB za pomocą CUDA.
 *
 * Operacje wykonywane przez funkcję:
 * 1. Alokacja pamięci GPU dla buforów wejściowego i wyjściowego.
 * 2. Kopiowanie danych z hosta do pamięci GPU.
 * 3. Uruchomienie kernela negativeKernel<<<...>>>().
 * 4. Synchronizacja i sprawdzenie błędów CUDA.
 * 5. Skopiowanie wyniku do bufora wyjściowego w pamięci hosta.
 * 6. Zwolnienie pamięci GPU.
 *
 * @param input       Dane wejściowe obrazu RGB (host).
 * @param output      Bufor wyjściowy RGB (host).
 * @param width       Szerokość obrazu (px).
 * @param height      Wysokość obrazu (px).
 * @param pitch       Pitch bufora wejściowego.
 */
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
