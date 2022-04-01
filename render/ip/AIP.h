#pragma once

/**
 * Average intensity projection.
 */
class AIP : public IP
{
public:
    using IP::IP;

    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        m_count++;
        m_sumIntensity += hitData.intensity;

        return true;
    }

    __device__ float4 getFinalColor() const
    {
        if(m_count == 0)
        {
            // évite de diviser par zéro
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            return fetchColor(m_sumIntensity / static_cast<float>(m_count));
        }
    }

private:
    long m_count = 0L;
    float m_sumIntensity = 0.0f;
};