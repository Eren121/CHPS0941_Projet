#pragma once

#include "IP.h"

/**
 * Local maximum intensity projection.
 */
class LMIP : public IP
{
public:
    __device__ LMIP(const RenderingTypeOptions& options)
        : IP(options), m_threshold(options.lmip.threshold) {}
    
    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        const float intensity = hitData.intensity;

        if(intensity >= m_threshold)
        {
            // On a atteint le seuil :
            // On rentre maintenant dans le mode de recherche du maximum local,
            // C'est-à-dire que simplement on stoppe dès que l'intensité diminue.
            m_thresholdHit = true;
        }

        if(m_thresholdHit && intensity < m_maxIntensity) {
            // Le maximum local est trouvé (à l'étape juste avant), on stoppe ici
            return false;
        }

        m_maxIntensity = max(intensity, m_maxIntensity);
        return true;
    }

    __device__ float4 getFinalColor() const
    {
        return fetchColor(m_maxIntensity);
    }

private:
    float m_threshold = 1.0f; // Quand thresold = 1.0f, LMIP == MIP
    bool m_thresholdHit = false;
    float m_maxIntensity = 0.0f;
};