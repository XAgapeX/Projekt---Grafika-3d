#pragma once

#include <cuda_runtime.h>

/**
 * @brief Zastosowanie filtra kontrastu na obrazie RGB za pomocą CUDA.
 *
 * Funkcja zmienia kontrast obrazu poprzez przesunięcie wartości pikseli względem
 * środka (128) i skalowanie przez podany współczynnik.
 *
 * @param input  Wskaźnik do danych wejściowych RGB (host memory)
 * @param output Wskaźnik do bufora wyjściowego RGB (host memory)
 * @param width  Szerokość obrazu w pikselach
 * @param height Wysokość obrazu w pikselach
 * @param inputPitch  Pitch bufora wejściowego w bajtach (ilość bajtów w jednej linii obrazu)
 * @param factor  Współczynnik kontrastu (>1 zwiększa kontrast, <1 zmniejsza kontrast)
 */
void applyContrastCUDA(unsigned char* input, unsigned char* output,
                       int width, int height, int inputPitch, float factor);
