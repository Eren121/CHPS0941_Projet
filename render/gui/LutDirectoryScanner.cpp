#include "LutDirectoryScanner.h"
#include "Gui.h"
#include <imgui.h>
#include <filesystem>

namespace fs = std::filesystem;

void LutDirectoryScanner::open(const std::string& dirName)
{
    m_dirName = dirName;
    m_background.load();

    m_isOpen = true;
    fs::path dirPath(dirName);

    // Liste non-récursivement tous les fichiers,
    // Cherche ceux qui se terminent en .lut,
    // Et essaye de charger une LUT avec ceux-ci.
    // Si le chargement réussi, on les ajoute à la liste des LUT chargées.

    for (const auto& dirEntry : fs::directory_iterator{dirPath}) 
    {
        const auto& path = dirEntry.path();

        if(dirEntry.is_regular_file() && path.extension() == ".lut") {
            
            Lut lut = Lut(path.string().c_str());
            lut.setUseLinearInterpolation(m_linearInterp);

            if(lut) {
                m_items.push_back(std::move(lut));
            }
        }
    }
}

bool LutDirectoryScanner::draw(Lut& outputLut, const RenderGui& gui)
{
    bool ret = false;

    if(m_isOpen) {

        if(m_linearInterp != gui.m_lutLinearInterp) {
            m_linearInterp = gui.m_lutLinearInterp;
            // Actualiser le mode pour chaque lut qui a changé
            for(Lut& lut : m_items) {
                lut.setUseLinearInterpolation(m_linearInterp);
            }
        }

        ImGui::SetNextWindowSize(ImVec2(520, 600), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("LUT", &m_isOpen))
        {
            {
                const ImVec4 color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
                ImGui::TextColored(color, "%d LUT trouvées dans '%s'.", static_cast<int>(m_items.size()), m_dirName.c_str());
            }

            int id = 0;
            for(const Lut& item : m_items)
            {
                ImGui::PushID(id); id++;

                ImGui::Text("%s (%d values)", item.getFilePath().c_str(), item.getTextureSize().x);

                // Affiche une preview de la LUT
                const ImTextureID textureID = reinterpret_cast<ImTextureID>(item.getPreviewTextureID());
                const ImTextureID bgTexID = reinterpret_cast<ImTextureID>(m_background.getTextureID());

                const int padding = 5;

                // Pas la taille des données de la texture, mais la taille que l'on souhaite en pixels
                const ImVec2 tex_size = ImVec2(ImGui::GetContentRegionAvail().x - padding * 2, 100);
                
                float delta = 0.0f;
                if(gui.m_correctLutTexRange) {
                    delta = 1.0f / (2.0f * item.getTextureSize().x);
                }

                const ImVec2 uv_min = ImVec2(delta, 0.0f); // Top-left
                const ImVec2 uv_max = ImVec2(1.0f - delta, 1.0f); // Lower-right
                const ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);   // No tint
                    
                const ImVec2 savedPos = ImGui::GetCursorPos();
                ImVec2 finalPosition;

                {
                    // Affiche le fond d'abord
                    // Puis on affiche la texture de gradient par dessus avec la même taille
                    // En réinitialisant la position du curseur
                    // Si le gradient possède des pixels transparents, ceux-ci seront mis en valeur
                    // Car on verra la texture de damier du fond (comme dans photoshop, gimp, etc)

                    // Indique combien de pixels prend 1 pattern de la texture de fond
                    // Utile pour que le background s'adapte à la taille de la fenêtre,
                    // il suffit d'une texture de quelques pixels
                    // et il se répète à l'infini
                    const float patternDensity = 10.0f;

                    const ImVec2 bg_size = ImVec2(tex_size.x + padding, tex_size.y + padding);
                    const ImVec2 bg_uv_min = ImVec2(0.0f, 0.0f);
                    const ImVec2 bg_uv_max = ImVec2(tex_size.x / patternDensity, tex_size.y / patternDensity);
                    ImVec4 bg_tint_col = tint_col;

                    if(!gui.m_highlightTransparency) {
                        bg_tint_col = ImVec4(0.0f, 0.0f, 0.0f, 1.0f); // Fond noir au lieu d'un damier
                    }

                    if(ImGui::ImageButton(bgTexID, tex_size, bg_uv_min, bg_uv_max, padding, ImVec4{0, 0, 0, 0}, bg_tint_col))
                    {
                        ret = true;
                        outputLut = item;
                    }
                    
                    finalPosition = ImGui::GetCursorPos();
                }

                const ImVec4 border_col = ImVec4(0, 0, 0, 0);

                // Décale pour être synchronisé avec les bordures
                ImGui::SetCursorPos({savedPos.x + padding, savedPos.y + padding});
                ImGui::Image(textureID, tex_size, uv_min, uv_max, tint_col, border_col);

                // Replace le curseur à la fin du bouton qui englobe tout
                ImGui::SetCursorPos(finalPosition);

                ImGui::PopID();
            }
        }

        ImGui::End();
    }
    
    return ret;
}