#include "sobel.cuh"
#include <cuda_runtime.h>
#include <math.h>

typedef unsigned char uchar;

/**
 * @brief Stałe maski Sobela dla osi X.
 *
 * Używane do obliczania gradientu w poziomie.
 */
__constant__ int d_sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

/**
 * @brief Stałe maski Sobela dla osi Y.
 *
 * Używane do obliczania gradientu w pionie.
 */
__constant__ int d_sobelY[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

/**
 * @brief Oblicza luminancję piksela RGB (przekształcenie na Y).
 *
 * @param p wskaźnik na 3 bajty (R,G,B)
 * @return luminancja z zakresu 0–255
 */
__device__
inline float luminance(const uchar* p)
{
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
}

/**
 * @brief Kernel CUDA wykonujący filtr Sobela w obrazie RGB.
 *
 * Kernel:
 * - pobiera piksel wejściowy,
 * - oblicza gradient Gx i Gy,
 * - wylicza moduł gradientu,
 * - zapisuje wynik jako piksel grayscale (R=G=B),
 * - obsługuje piksele brzegowe (clamping).
 *
 * @param input obraz wejściowy RGB888 w pamięci GPU
 * @param output obraz wyjściowy RGB888
 * @param width szerokość obrazu
 * @param height wysokość obrazu
 * @param pitch liczba bajtów jednego wiersza obrazu
 */
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

    // 3x3 okno Sobela
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

    // Zapisujemy jako RGB grayscale
    out[0] = edge;
    out[1] = edge;
    out[2] = edge;
}

/**
 * @brief Host-funkcja uruchamiająca CUDA Sobel.
 *
 * Etapy:
 * 1. alokacja pamięci GPU,
 * 2. kopiowanie obrazu wejściowego (CPU → GPU),
 * 3. uruchomienie kernela sobelKernel,
 * 4. kopiowanie wyników (GPU → CPU),
 * 5. zwolnienie pamięci GPU.
 *
 * @param input bufor wejściowy RGB888
 * @param output bufor wyjściowy RGB888
 * @param width szerokość obrazu
 * @param height wysokość obrazu
 * @param pitch liczba bajtów jednego wiersza
 */
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
