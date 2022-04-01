#pragma once

#include "IP.h"

/**
 * Minimum intensity projection.
 */
class MinIP : public IP
{
public:
    using IP::IP;
    
    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        m_minIntensity = min(hitData.intensity, m_minIntensity);
        
        if(m_minIntensity == 0.0f)
        {
            // Pas la peine de continuer, intensité min. atteinte
            return false;
        }
        else
        {
            return true;
        }
    }

    __device__ float4 getFinalColor() const
    {
        return fetchColor(m_minIntensity);
    }

private:
    float m_minIntensity = 1.0f;
};