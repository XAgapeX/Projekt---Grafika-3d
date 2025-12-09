#include "grayscale.cuh"
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    return; \
}

/**
 * @brief Kernel CUDA konwertujący RGB do YCbCr dla każdego piksela.
 *
 * Wzory konwersji:
 *   Y  = 0.299*R + 0.587*G + 0.114*B
 *   Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
 *   Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
 *
 * @param input     Bufor wejściowy RGB (GPU memory).
 * @param inputPitch Pitch bufora wejściowego.
 * @param outputY   Bufor wyjściowy Y (GPU memory).
 * @param outputCb  Bufor wyjściowy Cb (GPU memory).
 * @param outputCr  Bufor wyjściowy Cr (GPU memory).
 * @param width     Szerokość obrazu w pikselach.
 * @param height    Wysokość obrazu w pikselach.
 */
__global__ void grayscaleKernel(unsigned char* input, int inputPitch,
                                unsigned char* outputY, unsigned char* outputCb, unsigned char* outputCr,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * inputPitch + x * 3;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        float Y  = 0.299f * r + 0.587f * g + 0.114f * b;
        float Cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;
        float Cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;

        outputY[y * width + x] = static_cast<unsigned char>(Y);
        outputCb[y * width + x] = static_cast<unsigned char>(Cb);
        outputCr[y * width + x] = static_cast<unsigned char>(Cr);
    }
}

/**
 * @brief Funkcja nakładająca filtr grayscale (RGB -> YCbCr) za pomocą CUDA.
 *
 * Operacje wykonywane przez funkcję:
 * 1. Alokacja pamięci GPU dla buforów wejściowego (RGB) i wyjściowych (Y, Cb, Cr).
 * 2. Kopiowanie danych z hosta do pamięci GPU.
 * 3. Uruchomienie kernela grayscaleKernel<<<...>>>().
 * 4. Synchronizacja i sprawdzenie błędów CUDA.
 * 5. Skopiowanie wyników do buforów wyjściowych w pamięci hosta.
 * 6. Zwolnienie pamięci GPU.
 *
 * @param input      Dane wejściowe obrazu RGB (host).
 * @param outputY    Bufor wyjściowy Y (host).
 * @param outputCb   Bufor wyjściowy Cb (host).
 * @param outputCr   Bufor wyjściowy Cr (host).
 * @param width      Szerokość obrazu (px).
 * @param height     Wysokość obrazu (px).
 * @param inputPitch Pitch bufora wejściowego.
 */
void applyGrayscaleCUDA(unsigned char* input,
                        unsigned char* outputY,
                        unsigned char* outputCb,
                        unsigned char* outputCr,
                        int width, int height, int inputPitch)
{
    unsigned char *d_input, *d_outputY, *d_outputCb, *d_outputCr;
    size_t graySize = width * height;
    size_t inputSize = inputPitch * height;

    cudaError_t err;

    err = cudaMalloc(&d_input, inputSize);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_outputY, graySize);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_outputCb, graySize);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_outputCr, graySize);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    grayscaleKernel<<<blocks, threads>>>(d_input, inputPitch, d_outputY, d_outputCb, d_outputCr, width, height);

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    err = cudaMemcpy(outputY, d_outputY, graySize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
    err = cudaMemcpy(outputCb, d_outputCb, graySize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
    err = cudaMemcpy(outputCr, d_outputCr, graySize, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    cudaFree(d_input);
    cudaFree(d_outputY);
    cudaFree(d_outputCb);
    cudaFree(d_outputCr);
}
