#include "Lut.h"
#include <fstream>
#include <utility>
#include <iostream>
#include <GL/glew.h>

Lut::Lut(const char *lutFilePath)
    : m_filePath(lutFilePath)
{
    std::ifstream input(lutFilePath);

    printf("Reading lut '%s'\n", lutFilePath);

    if(input)
    {
        // Si on lit des chars, std::ifstream l'interprète comme caractère par caractère
        // On doit donc d'abord lire en int puis convertir en char

        uint4 rgba = {};

        while(input >> rgba.x >> rgba.y >> rgba.z >> rgba.w) {
            m_gradient.push_back(make_uchar4(rgba.x, rgba.y, rgba.z, rgba.w));
        }

        // On vérifie qu'on a lu l'entiereté du fichier à la fin
        if(input.eof()) {
            m_fileReadOK = true;
        }
    }

    if(!m_fileReadOK) {
        // Si on a pas réussi à lire le fichier,
        // On fournit une LUT basique par défaut (fonction identité)
        // En réalité c'est légèrement décalé car chaque valeur donne le centre des pixels et non les bords
        // Mais on ne voit pas la différence avec un oeil humain
        
        m_gradient = {};
        for(int i = 0; i < 256; i++) {
            m_gradient.push_back(make_uchar4(i, i, i, i));
        }
    }

    loadPreviewTexture();
    m_cudaTexture = TextureCuda(m_gradient.data(), m_gradient.size());
}

Lut::~Lut()
{
    glDeleteTextures(1, &m_textureID);
    m_textureID = 0;
}

Lut::Lut(const Lut& other)
    : m_filePath(other.m_filePath),
      m_fileReadOK(other.m_fileReadOK),
      m_gradient(other.m_gradient),
      m_interpolate(other.m_interpolate)
{
    loadPreviewTexture();
    m_cudaTexture = TextureCuda(m_gradient.data(), m_gradient.size());
}

Lut& Lut::operator=(Lut other)
{
    swap(*this, other);
    return *this;
}

Lut::Lut(Lut&& other)
{
    swap(*this, other);
}

void swap(Lut& left, Lut& right)
{
    using std::swap;
    swap(left.m_fileReadOK, right.m_fileReadOK);
    swap(left.m_gradient, right.m_gradient);
    swap(left.m_filePath, right.m_filePath);
    swap(left.m_textureID, right.m_textureID);
    swap(left.m_cudaTexture, right.m_cudaTexture);
    swap(left.m_interpolate, right.m_interpolate);
}

void Lut::loadPreviewTexture()
{
    glGenTextures(1, &m_textureID);

    glBindTexture(GL_TEXTURE_2D, m_textureID);

    // Options de filtrage: on clampe les couleurs aux bords pour le gradient,
    // et on voit de l'interpolation linéaire
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    setUseLinearInterpolation(m_interpolate);

    // Envoie les données du gradient à la texture n x 1 où n est le nombre de couleurs du gradient
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_gradient.size(), 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_gradient.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Lut::setUseLinearInterpolation(bool linear)
{
    m_interpolate = linear;

    glBindTexture(GL_TEXTURE_2D, m_textureID);

    if(linear) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
}