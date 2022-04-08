#pragma once

#include "../helper_cuda.h"
#include "../RenderingTypeOption.h"

/**
 * Données transmises aux IPs lors de la collision avec un Voxel.
 */
struct VoxelHitData
{
    __device__ VoxelHitData(const RenderingTypeOptions& options)
        : options(options),
          lut(options.lut)
    {
    }

    const RenderingTypeOptions& options; // Les options du GUI
    const LutData& lut; // Table des couleurs (intensité => couleur)
    float intensity; // L'intensité du voxel courant

    vec3f current_pos; // Position courante du voxel dans le monde
};