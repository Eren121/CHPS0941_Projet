#include "Gui.h"
#include "../common/ScreenDisplay.h"
#include <imgui.h>
#include <fstream>
#include <filesystem>
#include <functional>

namespace fs = std::filesystem;

RenderGui::RenderGui()
 : m_renderingTypeID(static_cast<int>(RENDER_MIP))
{
}

void RenderGui::init(ScreenDisplay& display)
{
    m_defaultCamera.camera = display.m_camera;
    m_defaultCamera.translateCamera = display.translateCamera;

    m_lutBackground.load();

    // Initialise la LUT avec la LUT par défaut
    // Normalement le fichier existe
    // S'il n'existe pas c'est pas grave,
    // l'application ne crashe pas et indique une erreur de lecture dans le gui
    tryLoadLut("../../data/lut/fusion.lut");
}

void RenderGui::tryLoadLut(const char *pathName)
{
    strncpy(m_lutPath, pathName, sizeof(m_lutPath));
    m_lut = Lut(m_lutPath);
    m_lut.setUseLinearInterpolation(m_lutLinearInterp);
    m_lutPathType = getPathType(pathName);
}

void RenderGui::draw(ScreenDisplay& display, LaunchParams& params)
{
    RenderingTypeOptions& options = params.renderingTypeOptions;

    // Affiche les options pour chaque mode de rendu

    if(ImGui::CollapsingHeader("Intensity projection"))
    {
        std::function<ImGuiTabItemFlags(RenderingType)> itemFlagsOf = [](RenderingType type) {
            return ImGuiTabItemFlags_None;
        };

        for(int i = 0; i < RENDER_Count; i++)
        {
            if(ImGui::RadioButton(renderingTypeNames[i], &m_renderingTypeID, i)) {
                
                // Quand l'utilisateur clic sur un mode,
                // on ouvre automatiquement l'onglet d'options de ce mode
                itemFlagsOf = [i](RenderingType type) {
                    return type == i ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
                };
            }
        }

        const ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None | ImGuiTabBarFlags_TabListPopupButton;
        if(ImGui::BeginTabBar("Options", tab_bar_flags))
        {
            if(ImGui::BeginTabItem("LMIP", nullptr, itemFlagsOf(RENDER_LMIP)))
            {
                ImGui::SliderFloat("threshold", &options.lmip.threshold, 0.0f, 1.0f);
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("DEMIP", nullptr, itemFlagsOf(RENDER_DEMIP))) {
                
                ImGui::PushItemWidth(-ImGui::GetContentRegionAvail().x * 0.5f);
                ImGui::SliderFloat("Depth weight", &options.demip.dw, 0.0f, 1.0f);
                ImGui::PopItemWidth();
                
                const int depthPercent = static_cast<int>((1.0f - options.demip.dw) * 100.0f);
                const ImVec4 color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                ImGui::SameLine();
                ImGui::TextColored(color, "%d%% mip, %d%% occlusion", depthPercent, 100 - depthPercent);

                ImGui::SliderFloat("Tolérance", &options.demip.tolerance, 0.0f, 1.0f);
                ImGui::Checkbox("Show closest voxel depth only", &options.demip.showDepthOnly);
                ImGui::EndTabItem();
            }
            if(ImGui::BeginTabItem("MIDA", nullptr, itemFlagsOf(RENDER_MIDA))) {
                
                float& gamma = options.mida.gamma;
                const float gamma_min = -1.0f;
                const float gamma_max = 0.0f;

                ImGui::PushItemWidth(-ImGui::GetContentRegionAvail().x * 0.5f);
                ImGui::SliderFloat("Gamma", &gamma, gamma_min, gamma_max);
                ImGui::PopItemWidth();
                
                const int midaPercent = static_cast<int>(-gamma * 100.0f);
                const ImVec4 color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                ImGui::SameLine();
                ImGui::TextColored(color, "%d%% dvr, %d%% mida", midaPercent, 100 - midaPercent);

                ImGui::EndTabItem();
            }
            
            ImGui::EndTabBar();
        }
    }

    // Affiche les options pour la LUT

    if(ImGui::CollapsingHeader("Table de couleurs"))
    {
        // Commencer par "##" cache le label
        if(ImGui::InputTextWithHint("##Fichier de LUT", "Entrez le chemin de la lut (.lut)", m_lutPath, IM_ARRAYSIZE(m_lutPath)))
        {
            // Condition effectuée quand l'utilisateur modifie le chemin
            m_lutPathType = getPathType(m_lutPath);
        }

        const bool hasLutPathChangedSinceLoad = (m_lutPath != m_lut.getFilePath());

        // Partie pour charger la LUT
        {
            if(m_lutPathType != File) {
                ImGui::BeginDisabled();
            }

            // Permet d'avoir un label dynamique avec le même ID (pour que le bouton marche)
            // "###" sépare le label affiché / l'ID interne
            const char* const reloadLabel = hasLutPathChangedSinceLoad ?
                "Charger le fichier de LUT###charger_lut" :
                "Re-charger le fichier###charger_lut";
            
            if(ImGui::Button(reloadLabel)) {
                tryLoadLut(m_lutPath);
            }

            if(m_lutPathType != File) {

                ImGui::EndDisabled();
            }
        }

        // Partie pour scanner le répertoire
        {
            ImGui::SameLine();
            if(m_lutPathType == DontExist) {
                ImGui::BeginDisabled();
            }
            if(ImGui::Button("Scanner le répertoire")) {

                // Si c'est un fichier, scanner le répertoire parent
                if(m_lutPathType == File) {
                    m_lutPathType = Directory;

                    std::string parentPath = fs::path{m_lutPath}.parent_path().string();
                    strncpy(m_lutPath, parentPath.c_str(), sizeof(m_lutPath));
                }

                m_lutScanner.open(m_lutPath);
            }
            if(m_lutPathType == DontExist) {
                ImGui::EndDisabled();
            }
        }

        if(!hasLutPathChangedSinceLoad && m_lutPath != std::string("")) {
            ImGui::SameLine();

            if(!m_lut) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Erreur de lecture");
            }
            else {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Fichier lu avec succès");
            }
        }

        // Affiche une preview de la LUT
        const ImTextureID textureID = reinterpret_cast<ImTextureID>(m_lut.getPreviewTextureID());
        const ImTextureID bgTexID = reinterpret_cast<ImTextureID>(m_lutBackground.getTextureID());

        // Pas la taille des données de la texture, mais la taille que l'on souhaite en pixels
        const ImVec2 tex_size = ImVec2(ImGui::GetContentRegionAvail().x, 100);
        
        const ImVec2 uv_min = ImVec2(options.lut.correctTexCoord(0.0f), 0.0f); // Top-left
        const ImVec2 uv_max = ImVec2(options.lut.correctTexCoord(1.0f), 1.0f); // Lower-right
        const ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);   // No tint
        const ImVec4 border_col = ImVec4(1.0f, 1.0f, 1.0f, 0.5f); // 50% opaque white

        {
            // Affiche le fond d'abord
            // Puis on affiche la texture de gradient par dessus avec la même taille
            // En réinitialisant la position du curseur
            // Si le gradient possède des pixels transparents, ceux-ci seront mis en valeur
            // Car on verra la texture de damier du fond (comme dans photoshop, gimp, etc)
            const ImVec2 cursorPos = ImGui::GetCursorPos();

            // Indique combien de pixels prend 1 pattern de la texture de fond
            // Utile pour que le background s'adapte à la taille de la fenêtre,
            // il suffit d'une texture de quelques pixels
            // et il se répète à l'infini
            const float patternDensity = 10.0f;

            const ImVec2 bg_uv_min = ImVec2(0.0f, 0.0f);
            const ImVec2 bg_uv_max = ImVec2(tex_size.x / patternDensity, tex_size.y / patternDensity);
            ImVec4 bg_tint_col = tint_col;

            if(!m_highlightTransparency) {
                bg_tint_col = ImVec4(0.0f, 0.0f, 0.0f, 1.0f); // Fond noir au lieu d'un damier
            }

            ImGui::Image(bgTexID, tex_size, bg_uv_min, bg_uv_max, bg_tint_col, border_col);
            ImGui::SetCursorPos(cursorPos);
        }

        ImGui::Image(textureID, tex_size, uv_min, uv_max, tint_col, border_col);

        if(ImGui::Checkbox("Interpolation linéaire", &m_lutLinearInterp)) {
            m_lut.setUseLinearInterpolation(m_lutLinearInterp);
        }

        ImGui::Checkbox("Mettre en valeur la transparence", &m_highlightTransparency);

        ImGui::Checkbox("Uniformiser le gradient", &m_correctLutTexRange);
    }

    if(ImGui::CollapsingHeader("Caméra"))
    {
        if(ImGui::Button("Réinitialiser la caméra")) {
            display.m_camera = m_defaultCamera.camera;
            display.translateCamera = m_defaultCamera.translateCamera;
        }
    }

    if(m_lutScanner.draw(m_lut, *this)) {
        strncpy(m_lutPath, m_lut.getFilePath().c_str(), sizeof(m_lutPath));
        m_lutPathType = getPathType(m_lutPath);
    }

    params.renderingType = static_cast<RenderingType>(m_renderingTypeID);
    options.lut.texture = m_lut.getCudaTextureObject();

    if(m_correctLutTexRange) {
        const float delta = 1.0f / (2.0f * m_lut.getTextureSize().x);
        options.lut.texCoordRange.min = delta;
        options.lut.texCoordRange.max = 1.0f - delta;
    }
    else {
        options.lut.texCoordRange.min = 0.0f;
        options.lut.texCoordRange.max = 1.0f;
    }
}

RenderGui::PathType RenderGui::getPathType(const char *pathName)
{
    // On doit utiliser l'overload sinon cela throw des exceptions
    // Dans certains cas, Windows refuse et throw, par ex. le fichier "con" (console) est réservé
    // si on appelle fs::exists("C:/dir/con") par ex. cela throw.

    std::error_code code;

    fs::path path(pathName);

    if(!fs::exists(path, code)) {
        return DontExist;
    }
    if(fs::is_regular_file(path, code)) {
        return File;
    }
    else if(fs::is_directory(path, code)) {
        return Directory;
    }
    else {
        return DontExist; // Fichiers spéciaux / erreur, on considère qu'ils n'existent pas
    }
}