#pragma once

void applyCartoonCUDA(unsigned char* input, unsigned char* output,
                      int width, int height, int inputPitch,
                      int colorLevels = 8, float edgeThreshold = 50.0f);


