#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <string>
#include <cstdlib>
#include <stdint.h>
#include <iostream>
#include "stb_image.h"

struct color_t{
    unsigned char r,g,b;
};

struct image_t{
    int width,height,bpp;
    unsigned char *data;

    void loadImage(const std::string &path);
    void createWhiteImage(const int w, const int h);
    void freeImage();
};
#endif