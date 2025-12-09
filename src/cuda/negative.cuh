#pragma once

/**
 * @brief Zastosowanie filtra negatywnego (inwersji kolorów) na obrazie RGB za pomocą CUDA.
 *
 * Funkcja przyjmuje bufor wejściowy (RGB) oraz bufor wyjściowy i odwraca wartości
 * kolorów dla każdego piksela: R -> 255-R, G -> 255-G, B -> 255-B.
 *
 * @param input       Dane wejściowe obrazu RGB (host memory).
 * @param output      Bufor wyjściowy, do którego zostanie zapisany obraz po filtrze.
 * @param width       Szerokość obrazu w pikselach.
 * @param height      Wysokość obrazu w pikselach.
 * @param inputPitch  Ilość bajtów w jednej linii obrazu (pitch) w buforze wejściowym.
 *
 * @note Funkcja sama alokuje pamięć GPU, kopiuje dane, uruchamia kernel,
 *       synchronizuje, kopiuje wynik i zwalnia pamięć.
 */
void applyNegativeCUDA(unsigned char* input, unsigned char* output,
                    int width, int height, int inputPitch);
