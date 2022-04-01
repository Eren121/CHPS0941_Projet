#pragma once

#include "IP.h"

/**
 * La couleur est celle de la profondeur de l'intensité du premier voxel rencontré
 * par le rayon (La profondeur est dans [0;1] puis convertie en couleur par la LUT).
 */
class DepthOnly : public IP
{
public:
    using IP::IP;

    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        // Termine directement dès le premier voxel rencontré,
        // et on stocke sa profondeur
        m_depth = hitData.depth;
        return false;
    }

    __device__ float4 getFinalColor() const
    {
        return fetchColor(m_depth);
    }

private:
    float m_depth = 0.0f;
};