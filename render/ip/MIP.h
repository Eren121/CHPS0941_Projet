#pragma once

#include "IP.h"
#include "../helper_math.h"

/**
 * Maximum intensity projection.
 */
class MIP : public IP
{
public:
    using IP::IP;

    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        m_maxIntensity = max(hitData.intensity, m_maxIntensity);
        
        if(m_maxIntensity == 1.0f)
        {
            // Pas la peine de continuer, intensité max. atteint
            return false;
        }
        else
        {
            return true;
        }
    }

    __device__ float4 getFinalColor() const
    {
        return fetchColor(m_maxIntensity);
    }

private:
    float m_maxIntensity = 0.0f; // éviter d'utiliser FLT_MIN/FLT_MAX, qui peut ne pas être valide sur le device
};