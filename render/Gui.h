#pragma once

#include "RenderingType.h"

/**
 * Contient les données pour le menu pour le rendu MIP.
 * Pour se souvenir de la valeur précédente dans imgui, il faut utiliser la même variable
 * à chaque fois. Pour éviter de polluer l'espace global de variables locales statiques,
 * on agglomère tout dans une classe.
 */
class RenderGui
{
public:
    RenderGui();

    /**
     * Dessiner le GUI dans son propre onglet.
     */
    void draw();

    RenderingType getRenderingType() const
    { return static_cast<RenderingType>(m_renderingTypeID); }

private:
    int m_renderingTypeID;
};