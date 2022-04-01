#pragma once

#include <string>
#include <vector>
#include "../Lut.h"
#include "../CheckerboardTexture.h"

class RenderGui;

/**
 * Scanne un répertoire et cherche tous les fichiers .lut pour les afficher dans une fenêtre.
 */
class LutDirectoryScanner
{
public:
    /**
     * Ouvre la fenêtre des LUTs
     * et scanne le répertoire qui contient les fichiers de lut (non récursif).
     */
    void open(const std::string& directory);

    /**
     * @return true Si l'utilisateur a cliqué sur une LUT pour la charger, stocké dans l'argument lut,
     * Dans ce cas retourne le chemin (on peut se permettre de la recharger, les fichiers sont légers).
     */
    bool draw(Lut& outputLut, const RenderGui& gui);

private:
    std::string m_dirName;
    bool m_isOpen = false;
    bool m_linearInterp = true;

    std::vector<Lut> m_items;
    CheckerboardTexture m_background;
};