#ifndef RENDERINGTYPEOPTION_H
#define RENDERINGTYPEOPTION_H

#include "RenderingType.h"
#include "helper_math.h"

/**
 * Table des couleurs (côté GPU).
 * Gradient de gauche à droite.
 * tex2D<float4>(lut, 0.0, y) = couleur rgba (float4) de l'intensité 0.0
 * tex2D<float4>(lut, 1.0, y) = couleur rgba (float4) de l'intensité 1.0
 * La coordonnée en Y ne change pas le résultat.
 *
 * @remarks
 * La variable texCoordRange est là pour corriger les UVs pour les problèmes de gradients
 * quand peu de pixels sont utilisés. La mettre à [0;1] ne change rien.
 * Par ex. quand il y a seulement 2 pixels, l'interpolation doit commencer à 0.25 et finir à 0.75
 * à cause de comment fonctionne les coordonnées de texture.
 *
 * Les fichiers de LUT fournis ont tous 256 valeurs, donc la différence est inperceptible ici
 * mais j'ai voulu pouvoir utiliser des fichiers avec aussi peu de valeurs que l'on veut
 * pour voir le résultat simplement.
 */
struct LutData
{
    cudaTextureObject_t texture;

    struct {
        float min = 0.0f;
        float max = 1.0f;
    } texCoordRange;

    __device__ __host__
    float correctTexCoord(float x) const {
        return unNormalize(x, texCoordRange.min, texCoordRange.max);
    }

};

/**
 * agglomère toutes les options pour tous les types de rendus.
 */
struct RenderingTypeOptions
{
    LutData lut;

    struct
    {
        /**
         * La tolérance pour la similarité entre deux matériau.
         * Dans [0;1].
         *
         * Si vaut 1, alors n'importe quel matériau sera toujours similaire à un autre.
         * Si vaut 0, on cherche l'exact même autre matériau.
         */
        float threshold = 1.0f;
    } lmip;

    struct
    {
        bool showDepthOnly = false;
        float dw = 0.15f; // dans [0;1]; Valeur par défaut suggérée dans la publication
        float tolerance = 0.15f; // dans [0;1]; Tolérance pour la similarité entre deux matériaux
    } demip;

    struct
    {
        // Transitionner doucement entre rendu DVR et MIDA
        // in [-1;0]
        // gamma == -1: le rendu est DVR
        // gamma == 0: le rendu est MIDA
        float gamma = 0.0f;
    } mida;
};

#endif /* RENDERINGTYPEOPTION_H */
