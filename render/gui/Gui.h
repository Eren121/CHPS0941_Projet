#pragma once

#include <string>
#include "LutDirectoryScanner.h"
#include "../RenderingType.h"
#include "../Lut.h"
#include "../CheckerboardTexture.h"
#include "../../common/LaunchParams.h"
#include "../../common/camera.h"

class ScreenDisplay;
class RenderGui;

/**
 * Contient les données pour le menu pour le rendu MIP.
 * Pour se souvenir de la valeur précédente dans imgui, il faut utiliser la même variable
 * à chaque fois. Pour éviter de polluer l'espace global de variables locales statiques,
 * on agglomère tout dans une classe.
 */
class RenderGui
{
    friend class LutDirectoryScanner;

private:
    enum PathType {
        DontExist, File, Directory
    };

public:
    RenderGui();

    /**
     * @brief Initialise la classe.
     * La raison d'existence de cette méthode est que lors du constructeur, glewInit() n'est pas encore appelé et on
     * ne peut donc pas utiliser OpenGL.
     */
    void init(ScreenDisplay& display);

    /**
     * Dessiner le GUI dans son propre onglet et applique les options du GUI aux paramètres.
     */
    void draw(ScreenDisplay& display, LaunchParams& params);

    static PathType getPathType(const char *path);

private:
    int m_renderingTypeID = 0;

    void tryLoadLut(const char *pathName);

    char m_lutPath[1000] = {};
    PathType m_lutPathType = DontExist;

    LutDirectoryScanner m_lutScanner;

    /**
     * On veut évidemment interpoler pour le rendu,
     * mais ça peut être intéressant de voir le résultat quand ce n'est pas le cas
     * pour prévisualiser la LUT.
     */
    bool m_lutLinearInterp = true;

    /**
     * Si il faut mettre en valeur la transparence grâce au background
     * dans la preview.
     */
    bool m_highlightTransparency = true;

    bool m_correctLutTexRange = true;

    /**
     * Pour pouvoir réinitialiser la caméra.
     * Si on zoom trop, on arrive à une distance de 0.0 et il devient impossible de reset à la souris:
     * le GUI est là pour ça
     */
    struct {
        vec3f translateCamera;
        Camera camera;
    } m_defaultCamera;

    Lut m_lut;
    CheckerboardTexture m_lutBackground;
};