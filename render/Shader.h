#pragma once

#include <float.h>

namespace ip_type
{

/**
 * La méthode intensityProjection() du shader volume.cu accepte
 * une classe générique en argument qui permet d'implémenter une méthode d'IP
 * générique sans dupliquer le code.
 * 
 * Cette classe itère toutes les intensités de voxels rencontrés (du plus près au plus loin)
 * et retourne la valeur
 * correspondant à la méthode choisie suivant les voxels rencontrés.
 * 
 * On n'utilise pas l'héritage (une autre façon de faire qui aurait été possible)
 * car c'est délicat à utiliser avec CUDA quand on passe un pointeur d'une classe host -> device.
 *
 * Cette classe est juste un exemple et ne doit pas être utilisée, elle montre juste ce que doivent
 * contenir les fonctions.
 * 
 * Pour ce programme, on considère toujours le minimum absolu possible d'intensité à 0
 * et le maximum possible d'intensité à 1.
 * Cela simplifie car sinon pour initialiser à +/- Infinity, on ne pourrait pas utiliser
 * FLT_MAX/FLT_MIN, car cela n'est pas forcément identique pour du code device (seulement host).
 */
class BasicIP
{
public:
    /**
     * Constructeur & destructeur doivent être __device__.
     */
    __device__ BasicIP() {}
    __device__ ~BasicIP() {}

    /**
     * Appelé quand le prochain voxel a été rencontré.
     *
     * Peut stopper le rayon pour accélérer le rendu dans certains cas possibles,
     * par exemple si l'intensité maximale possible a été rencontrée pour le MIP,
     * cela ne sert à rien de parcourir les voxels restants car on a déjà au minimum
     * l'intensité maximum.
     * @param intensity L'intensité du voxel rencontrée
     * @return true - Pour continuer de boucler sur les prochains voxels.
     *         false - Pour indiquer que l'on peut directement terminer le rayon.
     */
    __device__ bool nextVoxelHit(float intensity)
    {
        return true;
    }

    /**
     * Appelé pour récupérer l'intensité finale du rayon après que tous les rayons aient
     * été calculés.
     */
    __device__ float getFinalIntensity() const
    {
        return 0.0f;
    }
};

class MIP
{
public:
    __device__ MIP() : m_maxIntensity(0.0f) {}
    __device__ ~MIP() {}

    __device__ bool nextVoxelHit(float intensity)
    {
        m_maxIntensity = max(intensity, m_maxIntensity);
        
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

    __device__ float getFinalIntensity() const
    {
        return m_maxIntensity;
    }

private:
    float m_maxIntensity;
};

class MinIP
{
public:
    __device__ MinIP() : m_minIntensity(1.0f) {}
    __device__ ~MinIP() {}

    __device__ bool nextVoxelHit(float intensity)
    {
        m_minIntensity = min(intensity, m_minIntensity);
        
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

    __device__ float getFinalIntensity() const
    {
        return m_minIntensity;
    }

private:
    float m_minIntensity;
};

class AIP
{
public:
    __device__ AIP() : m_count(0L), m_sumIntensity(0.0f) {}
    __device__ ~AIP() {}

    __device__ bool nextVoxelHit(float intensity)
    {
        m_count++;
        m_sumIntensity += intensity;

        return true;
    }

    __device__ float getFinalIntensity() const
    {
        if(m_count == 0)
        {
            // évite de diviser par zéro
            return 0.0f;
        }
        else
        {
            return m_sumIntensity / static_cast<float>(m_count);
        }
    }

private:
    long m_count;
    float m_sumIntensity;
};

}