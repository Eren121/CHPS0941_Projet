#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "TextureCuda.h"

/**
 * Lookup table pour les couleurs (côté CPU).
 * Permet de charger le fichier .lut, la texture OpenGL (pour la prévisualiser) ainsi que la texture CUDA.
 *
 * Pour gérer la texture CUDA, on peut soit utiliser l'interop. CUDA <=> OpenGL.
 * Ici, on utilisera les textures CUDA basiques, donc la texture sera deux fois sur le GPU (en OpenGL et en CUDA)
 * mais comme elles sont petites ce n'est pas grave.
 */
class Lut
{
public:
    /**
     * Créer une LUT invalide.
     */
    Lut() = default;

    /**
     * @param lutFilePath Le chemin de la lut (les fichiers .lut fournis).
     * Le format d'un fichier .lut est une suite de n lignes dont chaque ligne est composé de 4 entiers dans [0;255].
     * La première ligne indique la couleur pour l'intensité 0 et la dernière la couleur pour l'intensité 1, le reste étant
     * mappé dans [0;1] à intervalles réguliers.
     * Les couleurs qui ne sont pas indiquées sont déduites par interpolation linéaire entre les deux
     * valeurs les plus proches par les textures.
     *
     * <r> <g> <b> <a>
     * <r> <g> <b> <a>
     * ....
     *
     * @remarks
     * Si la lecture échoue, on y place une LUT basique (fonction indentité en niveau de gris).
     */
    Lut(const char *lutFilePath);
    ~Lut();

    Lut(const Lut&);
    Lut& operator=(Lut); // Both const Lut& and Lut&&
    Lut(Lut&&);

    /**
     * @return true Si la lecture du fichier de LUT s'est bien effectué, false sinon.
     */
    operator bool() const { return m_fileReadOK; }

    const std::string& getFilePath() const { return m_filePath; }
    unsigned int getPreviewTextureID() const { return m_textureID; }
    uint2 getTextureSize() const { return make_uint2(static_cast<unsigned int>(m_gradient.size()), 1); }
    cudaTextureObject_t getCudaTextureObject() const { return m_cudaTexture.getTextureObject(); }

    /**
     * Si la texture doit être GL_LINEAR ou GL_NEAREST. Par défaut, vaut GL_LINEAR.
     */
    void setUseLinearInterpolation(bool linear);

private:
    /**
     * Initialise m_textureID.
     */
    void loadPreviewTexture();

private:
    std::string m_filePath;
    bool m_fileReadOK = false;
    std::vector<uchar4> m_gradient;

    /**
     * ID OpenGL de la texture pour prévisualiser la LUT.
     * La texture est un gradient de 0.0 à gauche vers 1.0 à droite.
     */
    unsigned int m_textureID = 0;

    /**
     * La LUT en texture CUDA.
     */
    TextureCuda m_cudaTexture;

    bool m_interpolate = true;

    friend void swap(Lut& left, Lut& right);
};