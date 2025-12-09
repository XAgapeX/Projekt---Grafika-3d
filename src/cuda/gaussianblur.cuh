#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

typedef unsigned char uchar;

/**
 * @brief Zastosowanie filtra rozmycia Gaussa (Gaussian Blur) na obrazie RGB za pomocą CUDA.
 *
 * Filtr rozmywa obraz, stosując konwolucję z macierzą Gaussa (kernel 21x21, sigma=7),
 * co powoduje wygładzenie krawędzi i redukcję szumów.
 *
 * @param input  Wskaźnik do danych wejściowych RGB (host memory).
 * @param output Wskaźnik do bufora wyjściowego RGB (host memory).
 * @param width  Szerokość obrazu w pikselach.
 * @param height Wysokość obrazu w pikselach.
 * @param pitch  Pitch bufora wejściowego w bajtach (ilość bajtów w jednej linii obrazu).
 */
void applyGaussianBlurCUDA(const uchar* input, uchar* output,
                           int width, int height, int pitch);
