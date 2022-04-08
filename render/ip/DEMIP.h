#pragma once

#include "IP.h"


/**
 * Depth-enhanced intensity projection.
 *
 * On considère que deux matériaux sont similaires s'ils ont une opacité similaire
 * Si les LUT sont entièrement opaque, cela revient à chercher le voxel el plus proche de la caméra.
 *
 * Il faut d'abord tirer un rayon qui contiendra le matériau d'intensité maximal.
 */
class DEMIP : public IP
{
public:
    /**
     * @param mipMaterial Le matériau de l'intensité du rayon MIP tiré précédemment
     * @param mipDepth La profondeur du matériau rencontrée (dans [0;1])
     */
    __device__ DEMIP(const RenderingTypeOptions& options, float4 mipMaterial)
        : IP(options),
          m_tolerance(options.demip.tolerance),
          m_material(mipMaterial)
    {
    }

    __device__ ~DEMIP() = default;

    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        const float4 m = fetchColor(hitData.intensity);
        if(isMaterialSimilar(m)) {
            // On a trouvé le premier matériau similaire

            const float dw = hitData.options.demip.dw;
            const float depth = norme(point_in - hitData.current_pos) / norme(point_out - point_in);

            // color = MIPcolor ∗ (1−dw) +2 ∗ dw ∗ (1−depth)
            m_finalColor = m_material * (1.0f - dw) + 2.0f * dw * (1.0f - depth);

            // Bien attention à ne pas dépasser 0.0 et 1.0
            const float4 min = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 max = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            m_finalColor = clamp(m_finalColor, min, max);
            return false;
        }
        else {
            return true;
        }
    }

    __device__ float4 getFinalColor() const
    {
        return m_finalColor;
    }

private:
    __device__ bool isMaterialSimilar(const float4& m)
    {
        // On compare les opacités selon la tolérance
        return abs(m.w - m_material.w) <= m_tolerance;
    }

private:
    float m_tolerance;
    float4 m_material;
    float4 m_finalColor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
};