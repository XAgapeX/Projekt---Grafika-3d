#include "brightness.cuh"
#include <iostream>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " (at " << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    return; \
}

/**
 * @brief Kernel CUDA zwiększający lub zmniejszający jasność pikseli RGB.
 *
 * Równolegle przetwarza każdy piksel:
 * - dodaje wartość brightness do R, G, B,
 * - ogranicza wyniki do zakresu 0–255,
 * - zapisuje wynik do bufora wyjściowego.
 *
 * @param input wskaźnik do pamięci GPU z obrazem wejściowym
 * @param inputPitch liczba bajtów w jednym wierszu obrazu
 * @param output wskaźnik do pamięci GPU na obraz wyjściowy
 * @param width szerokość obrazu (px)
 * @param height wysokość obrazu (px)
 * @param brightness wartość jasności do dodania
 */
__global__ void brightnessKernel(unsigned char* input, int inputPitch,
                                 unsigned char* output,
                                 int width, int height, int brightness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * inputPitch + x * 3;

    int r = input[idx]     + brightness;
    int g = input[idx + 1] + brightness;
    int b = input[idx + 2] + brightness;

    r = max(0, min(255, r));
    g = max(0, min(255, g));
    b = max(0, min(255, b));

    output[idx]     = static_cast<unsigned char>(r);
    output[idx + 1] = static_cast<unsigned char>(g);
    output[idx + 2] = static_cast<unsigned char>(b);
}

/**
 * @brief Host-funkcja uruchamiająca filtrowanie jasności CUDA.
 *
 * Kolejne etapy:
 * 1. Alokacja pamięci GPU.
 * 2. Kopia obrazu wejściowego (CPU → GPU).
 * 3. Uruchomienie kernela brightnessKernel.
 * 4. Kopia wyników (GPU → CPU).
 * 5. Zwolnienie pamięci GPU.
 *
 * @param input bufor wejściowy RGB888 (host)
 * @param output bufor wyjściowy RGB888 (host)
 * @param width szerokość w pikselach
 * @param height wysokość w pikselach
 * @param inputPitch liczba bajtów jednego wiersza obrazu
 * @param brightness wartość jasności (-255 do 255)
 */
void applyBrightnessCUDA(unsigned char* input, unsigned char* output,
                         int width, int height, int inputPitch, int brightness)
{
    unsigned char *d_input, *d_output;
    size_t size = inputPitch * height;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    brightnessKernel<<<blocks, threads>>>(d_input, inputPitch, d_output,
                                          width, height, brightness);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
