#pragma once

#include "IP.h"

/**
 * Maximum Intensity Difference Accumulation.
 */
class MIDA : public IP
{
public:
    using IP::IP;

    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        const float delta = max(hitData.intensity - m_maxIntensityHit, 0.0f); // Dérivée de l'intensité

        // Facteur d'occlusion
        const float gamma = options.mida.gamma;
        const float beta = 1.0f - delta * (1.0f + min(gamma, 0.0f));
        
        // Couleur/opacité du voxel selon la LUT
        const float4 voxelMat = fetchColor(hitData.intensity);
        const float3 voxelCol = make_float3(voxelMat.x, voxelMat.y, voxelMat.z);
        const float voxelOpacity = voxelMat.w;

        // Sauvegarder le rayon
        const float3 prevColor = make_float3(m_material.x, m_material.y, m_material.z);
        const float prevOpacity = m_material.w;

        // Formule
        const float3 nextColor = beta * prevColor + (1.0f - beta * prevOpacity) * voxelOpacity * voxelCol;
		const float nextOpacity = beta * prevOpacity + (1.0f - beta * prevOpacity) * voxelOpacity;

        // Actualiser le rayon
        m_material = make_float4(nextColor.x, nextColor.y, nextColor.z, nextOpacity);

        // Actualiser l'intensité max.
        // A LA FIN (10min de debugging) car on l'utilise pour calculer delta
        m_maxIntensityHit = max(hitData.intensity, m_maxIntensityHit);
        
        return true;
    }

    __device__ float4 getFinalColor() const
    {
        return clampColor(m_material);
    }

private:
    // Couleur + opacité (= Matériau) finales du rayon
    // L'opacité n'est pas vraiment utilisé pour rendu du pixel
    // Mais sera utile pour actualiser la couleur selon l'algorithme MIDA à chaque voxel
    float4 m_material = {};

    // Intensité max. rencontrée jusque là
    float m_maxIntensityHit = 0.0f;
};