#pragma once

/**
 * @brief Nakłada filtr jasności na obraz RGB za pomocą CUDA.
 *
 * Funkcja:
 * - kopiuje dane wejściowe do pamięci GPU,
 * - uruchamia kernel zwiększający/zmniejszający jasność pikseli,
 * - kopiuje dane wynikowe z powrotem do RAM.
 *
 * @param input wskaźnik na bufor wejściowy (RGB888)
 * @param output wskaźnik na bufor wyjściowy (RGB888)
 * @param width szerokość obrazu w pikselach
 * @param height wysokość obrazu w pikselach
 * @param inputPitch liczba bajtów jednego wiersza obrazu (QImage::bytesPerLine)
 * @param brightness wartość jasności (-255 do 255)
 */
void applyBrightnessCUDA(unsigned char* input, unsigned char* output,
                         int width, int height, int inputPitch, int brightness);
