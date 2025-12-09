#include "sepia.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
    << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    return; \
}

/**
 * @brief Ogranicza wartość float do zakresu bajtowego [0, 255].
 *
 * @param x Wartość wejściowa.
 * @return unsigned char Wartość ucięta do dopuszczalnego zakresu.
 */
__device__ inline unsigned char clampToByte(float x) {
    if (x < 0.f) return 0;
    if (x > 255.f) return 255;
    return static_cast<unsigned char>(x);
}

/**
 * @brief Kernel CUDA nakładający efekt sepia na obraz RGB.
 *
 * Kernel działa dla każdego piksela niezależnie. Odcienie sepia powstają przez
 * liniową kombinację składowych RGB według wzorów:
 *
 *   R' = 0.393R + 0.769G + 0.189B
 *   G' = 0.349R + 0.686G + 0.168B
 *   B' = 0.272R + 0.534G + 0.131B
 *
 * @param input      Bufor wejściowy (RGB), pamięć GPU.
 * @param inputPitch Ilość bajtów w jednej linii wejściowego obrazu.
 * @param output     Bufor wyjściowy (RGB), pamięć GPU.
 * @param width      Szerokość obrazu w pikselach.
 * @param height     Wysokość obrazu w pikselach.
 */
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

/**
 * @brief Główna funkcja nakładająca filtr sepia na obraz za pomocą CUDA.
 *
 * Operacje wykonywane przez funkcję:
 * 1. Alokacja pamięci na GPU (input, output).
 * 2. Kopiowanie bufora RGB z hosta do pamięci GPU.
 * 3. Uruchomienie kernela sepiaKernel<<<...>>>().
 * 4. Synchronizacja i sprawdzenie błędów CUDA.
 * 5. Skopiowanie wyniku do bufora wyjściowego w pamięci hosta.
 * 6. Zwolnienie pamięci GPU.
 *
 * @param input       Dane wejściowe obrazu RGB (host).
 * @param output      Bufor wyjściowy RGB (host).
 * @param width       Szerokość obrazu (px).
 * @param height      Wysokość obrazu (px).
 * @param inputPitch  Pitch bufora wejściowego.
 */
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
