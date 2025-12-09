#ifndef SOBEL_CUH
#define SOBEL_CUH

#include <cuda_runtime.h>
#include <stdint.h>

typedef unsigned char uchar;

/**
 * @brief Uruchamia filtr wykrywania krawędzi Sobela (CUDA) na obrazie RGB.
 *
 * Funkcja:
 * - kopiuje dane wejściowe do pamięci GPU,
 * - uruchamia kernel Sobela,
 * - zwraca wynik jako obraz w odcieniach szarości (RGB identyczne),
 * - zwalnia pamięć GPU.
 *
 * @param input bufor wejściowy (RGB888)
 * @param output bufor wyjściowy (RGB888, wynik filtrowania)
 * @param width szerokość obrazu w pikselach
 * @param height wysokość obrazu w pikselach
 * @param pitch liczba bajtów jednego wiersza obrazu
 */
void applySobelCUDA(const uchar* input, uchar* output,
                    int width, int height, int pitch);

#endif
