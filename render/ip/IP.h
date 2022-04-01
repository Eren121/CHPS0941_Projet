#pragma once

#include "VoxelHitData.h"
#include "../../common/LaunchParams.h"
#include "../helper_math.h"

/**
 * Classe de base de toutes les IPs (intensity projection).
 *
 * Les classes filles doivent implémenter différentes fonctions.
 * On n'utilise pas de méthode virtuelle ce qui optimise les performance, et la virtualité
 * est mal supportée avec CUDA. Le shader appelera automatiquement différentes méthodes de l'IP.
 * Si les méthodes n'existent pas, la compilation échouera.
 * Voir la classe ExampleIP qui montre quelles fonctions doivent être implémentées.
 */
class IP
{
public:
    __device__ IP(const RenderingTypeOptions& options) : options(options) {}
    __device__ ~IP() = default;

    /**
     * Initialisé dans le shader OptiX.
     *
     * @remarks
     * On n'utilise pas le constructeur mais des setter pour éviter de surcharger le code et pour
     * plus de flexibilité.
     * 
     * @param lut 
     * @return __device__ 
     */
    __device__ void setLut(const LutData& lut) {
        this->lut = &lut;
    }
    
    /**
     * Donnée de sortie du shader, contient la profondeur maximale rencontrée.
     * Utilisé par la première passe de DEMIP.
     */
    float deepestDepthHit = 0.0f;

protected:
    /**
     * Récupère une couleur de la table de couleur suivant l'intensité
     * @param intensity Dans [0;1], intensité du voxel.
     * @return La couleur associée à l'intensité
     */
    __device__ float4 fetchColor(float intensity) const {
       return tex2D<float4>(lut->texture, lut->correctTexCoord(intensity), 0.0f); 
    }

    const RenderingTypeOptions& options;

private:
    const LutData* lut;
};