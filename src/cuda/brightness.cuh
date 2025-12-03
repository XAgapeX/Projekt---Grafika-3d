#pragma once


void applyBrightnessCUDA(unsigned char* input, unsigned char* output,
                        int width, int height, int inputPitch , int brightness);
