#pragma once
#include <cuda_runtime.h>

/**
 * @brief Zastosowanie filtra sepia na obrazie za pomocą CUDA.
 *
 * Funkcja wykonuje transformację kolorów RGB do odcieni sepii przy użyciu
 * kernela CUDA. Oczekuje bufora wejściowego (input), bufora wyjściowego (output),
 * oraz parametrów wymiarów obrazu.
 *
 * @param input       Dane wejściowe obrazu w formacie RGB (host memory).
 * @param output      Bufor wyjściowy, do którego zostanie zapisany obraz po filtrze.
 * @param width       Szerokość obrazu w pikselach.
 * @param height      Wysokość obrazu w pikselach.
 * @param inputPitch  Ilość bajtów w jednej linii obrazu (pitch) w buforze wejściowym.
 *
 * @note Funkcja sama alokuje pamięć na GPU, kopiuje dane, uruchamia kernel,
 *       synchronizuje, kopiuje wynik i zwalnia pamięć.
 */
void applySepiaCUDA(unsigned char* input, unsigned char* output,
                    int width, int height, int inputPitch);
