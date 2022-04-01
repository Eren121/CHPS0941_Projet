#ifndef TEXTURECUDA_H
#define TEXTURECUDA_H

#include "helper_cuda.h"

void testLut(cudaTextureObject_t lut);

/**
 * Wrapper de texture CUDA.
 * Ne gère que le cas : 1 pixel de haut sur N pixels de large,
 * car ici on veut gérer uniquement les LUT.
 */
class TextureCuda
{
public:
    TextureCuda() = default;

    /**
     * Créé une texture de valeurs dans [0;255] (équivalent GL_UNSIGNED_BYTE);
     *
     * @param data Pointeur vers une suite de de pixels (RGBA)
     * @param size Nombre de pixels
     */
    TextureCuda(const uchar4* data, size_t size);

    ~TextureCuda();

    TextureCuda(const TextureCuda&) = delete;
    TextureCuda& operator=(const TextureCuda&) = delete;

    TextureCuda(TextureCuda&&);
    TextureCuda& operator=(TextureCuda&& right);
    
    cudaTextureObject_t getTextureObject() const { return m_textureObj; }

private:
    cudaTextureObject_t m_textureObj = {};
    cudaArray_t m_d_array = {};

    friend void swap(TextureCuda& lhs, TextureCuda& rhs);
};

#endif /* TEXTURECUDA_H */
