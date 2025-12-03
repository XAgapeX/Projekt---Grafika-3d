#pragma once

void applyContrastCUDA(unsigned char* input, unsigned char* output,
                       int width, int height, int inputPitch, float factor);