#include "Gui.h"

#include <imgui.h>

RenderGui::RenderGui()
 : m_renderingTypeID(static_cast<int>(RENDER_MIP))
{}

void RenderGui::draw()
{
    const ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    if (ImGui::BeginTabBar("Rendu", tab_bar_flags))
    {
        for(int i = 0; i < RENDER_Count; i++)
        {
            ImGui::RadioButton(renderingTypeNames[i], &m_renderingTypeID, i);
        }

        ImGui::EndTabBar();
    }
}