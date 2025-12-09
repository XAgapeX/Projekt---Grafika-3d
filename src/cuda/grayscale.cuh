#pragma once
#include <cuda_runtime.h>

/**
 * @brief Zastosowanie filtra grayscale (przekształcenie RGB do YCbCr) za pomocą CUDA.
 *
 * Funkcja konwertuje każdy piksel obrazu RGB na przestrzeń YCbCr:
 *  - Y  = jasność (luminancja)
 *  - Cb = niebiesko-żółta składowa chrominancji
 *  - Cr = czerwono-zielona składowa chrominancji
 *
 * @param input       Dane wejściowe obrazu RGB (host memory).
 * @param outputY     Bufor wyjściowy Y (luminancja).
 * @param outputCb    Bufor wyjściowy Cb (chrominancja niebieska).
 * @param outputCr    Bufor wyjściowy Cr (chrominancja czerwona).
 * @param width       Szerokość obrazu w pikselach.
 * @param height      Wysokość obrazu w pikselach.
 * @param inputPitch  Ilość bajtów w jednej linii obrazu (pitch) w buforze wejściowym.
 *
 * @note Funkcja sama alokuje pamięć GPU, kopiuje dane, uruchamia kernel,
 *       synchronizuje, kopiuje wynik i zwalnia pamięć.
 */
void applyGrayscaleCUDA(unsigned char* input,
                        unsigned char* outputY,
                        unsigned char* outputCb,
                        unsigned char* outputCr,
                        int width, int height, int inputPitch);
