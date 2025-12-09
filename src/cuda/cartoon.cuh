#pragma once

#include <cuda_runtime.h>

/**
 * @brief Zastosowanie efektu rysunkowego ("cartoon") na obrazie RGB za pomocą CUDA.
 *
 * Filtr redukuje liczbę kolorów (posterization) i podkreśla krawędzie,
 * tworząc efekt podobny do rysunku kreskówkowego.
 *
 * @param input  Wskaźnik do danych wejściowych RGB (host memory)
 * @param output Wskaźnik do bufora wyjściowego RGB (host memory)
 * @param width  Szerokość obrazu w pikselach
 * @param height Wysokość obrazu w pikselach
 * @param inputPitch Pitch danych wejściowych (liczba bajtów w linii)
 * @param colorLevels Liczba poziomów kolorów do redukcji (domyślnie 8)
 * @param edgeThreshold Prog wykrywania krawędzi (domyślnie 50.0f)
 */
void applyCartoonCUDA(unsigned char* input, unsigned char* output,
                      int width, int height, int inputPitch,
                      int colorLevels = 8, float edgeThreshold = 50.0f);
