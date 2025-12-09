#include "gaussianblur.cuh"
#include <cuda_runtime.h>
#include <cmath>

typedef unsigned char uchar;

/**
 * @brief Rozmiar jądra Gaussa (kernel 21x21)
 */
#define KERNEL_SIZE 21
#define SIGMA 7.0f
#define RADIUS (KERNEL_SIZE / 2)

/**
 * @brief Stała pamięć GPU przechowująca wartości macierzy Gaussa
 */
__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

/**
 * @brief Bufor hosta do wypełnienia macierzy Gaussa przed przesłaniem na GPU
 */
static float h_kernel[KERNEL_SIZE * KERNEL_SIZE];

/**
 * @brief Generuje macierz Gaussa i przesyła ją do pamięci stałej GPU
 */
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

/**
 * @brief Jądro CUDA wykorzystujące pamięć współdzieloną do rozmycia Gaussa
 *
 * Wczytuje piksele bloku do pamięci współdzielonej, a następnie wykonuje konwolucję
 * dla każdego piksela z wykorzystaniem predefiniowanej macierzy Gaussa.
 *
 * @param input  Wskaźnik do danych wejściowych RGB
 * @param output Wskaźnik do danych wyjściowych RGB
 * @param width  Szerokość obrazu
 * @param height Wysokość obrazu
 * @param pitch  Pitch bufora wejściowego w bajtach
 */
__global__
void gaussianBlurSharedKernel(const uchar* input, uchar* output,
                              int width, int height, int pitch)
{
    extern __shared__ uchar s_data[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int sharedWidth = blockDim.x + 2 * RADIUS;
    int sharedHeight = blockDim.y + 2 * RADIUS;

    // Wczytywanie pikseli do pamięci współdzielonej
    for (int dy = ty; dy < sharedHeight; dy += blockDim.y) {
        for (int dx = tx; dx < sharedWidth; dx += blockDim.x) {
            int gx = blockIdx.x * blockDim.x + dx - RADIUS;
            int gy = blockIdx.y * blockDim.y + dy - RADIUS;

            gx = min(max(gx, 0), width - 1);
            gy = min(max(gy, 0), height - 1);

            uchar* dst = &s_data[(dy * sharedWidth + dx) * 3];
            const uchar* src = input + gy * pitch + gx * 3;

            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    // Konwolucja z jądrem Gaussa
    float r = 0, g = 0, b = 0;
    int idx = 0;

    for (int ky = -RADIUS; ky <= RADIUS; ++ky) {
        for (int kx = -RADIUS; kx <= RADIUS; ++kx) {
            int sx = tx + kx + RADIUS;
            int sy = ty + ky + RADIUS;

            uchar* pixel = &s_data[(sy * sharedWidth + sx) * 3];
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

/**
 * @brief Funkcja główna do zastosowania filtra Gaussian Blur
 *
 * Tworzy bufor GPU, kopiuje dane, wywołuje jądro CUDA i kopiuje wynik z powrotem na hosta.
 *
 * @param input  Dane wejściowe RGB (host memory)
 * @param output Bufor wyjściowy RGB (host memory)
 * @param width  Szerokość obrazu
 * @param height Wysokość obrazu
 * @param pitch  Pitch bufora wejściowego
 */
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

    dim3 block(16,16);
    dim3 grid((width + 15)/16, (height + 15)/16);
    size_t sharedMemSize = (block.x + 2*RADIUS) * (block.y + 2*RADIUS) * 3 * sizeof(uchar);

    gaussianBlurSharedKernel<<<grid, block, sharedMemSize>>>(d_input, d_output, width, height, pitch);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, totalSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
