#pragma once

#include "helper_cuda.h"

/**
 * Stocke une texture 2x2 en damier formée de deux couleurs.
 */
class CheckerboardTexture
{
public:
    CheckerboardTexture() = default;
    ~CheckerboardTexture();

    /**
     * @param color1, color2 Les couleurs utilisées pour faire le damier.
     * Par défaut ici sont celles utilisées par Photoshop.
     */
    void load(
        uchar4 color1 = make_uchar4(255, 255, 255, 255),
        uchar4 color2 = make_uchar4(204, 204, 204, 255));

    CheckerboardTexture(const CheckerboardTexture&) = delete;
    CheckerboardTexture& operator=(const CheckerboardTexture&) = delete;

    unsigned int getTextureID() const { return m_textureID; }

private:
    unsigned int m_textureID = 0;
};