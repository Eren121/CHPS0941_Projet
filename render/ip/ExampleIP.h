#include "IP.h"

/**
 * Cette classe est juste un exemple et ne doit pas être utilisée, elle montre juste ce que doivent
 * contenir les fonctions.
 */
class ExampleIP : public IP
{
public:
    using IP::IP;

    /**
     * Appelé quand le prochain voxel a été rencontré.
     *
     * Peut stopper le rayon pour accélérer le rendu dans certains cas possibles,
     * par exemple si l'intensité maximale possible a été rencontrée pour le MIP,
     * cela ne sert à rien de parcourir les voxels restants car on a déjà au minimum
     * l'intensité maximum.
     * @param intensity L'intensité du voxel rencontrée
     * @return true - Pour continuer de boucler sur les prochains voxels.
     *         false - Pour indiquer que l'on peut directement terminer le rayon
     *         Pour l'optimisation.
     */
    __device__ bool nextVoxelHit(const VoxelHitData& hitData)
    {
        return true;
    }

    /**
     * Appelé pour récupérer la couleur finale du rayon.
     */
    __device__ float4 getFinalColor() const
    {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
};