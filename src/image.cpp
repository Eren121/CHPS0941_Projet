#define STB_IMAGE_IMPLEMENTATION
#include "image.h"

void image_t::loadImage(const std::string &path){
    data = stbi_load(path.c_str(), &width, &height, &bpp, 4);
}
void image_t::createWhiteImage(const int w, const int h){;
    width = w;
    height = h;
    bpp = 3;
    data = (unsigned char*)malloc(width*height*4);
    memset(data,255,w*h*3);
}
void image_t::freeImage(){
    stbi_image_free(data);
}
