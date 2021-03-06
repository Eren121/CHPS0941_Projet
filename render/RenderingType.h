#pragma once

enum RenderingType {
    RENDER_MIP,
    RENDER_MINIP,
    RENDER_AIP,

    RENDER_LMIP,
    RENDER_DEMIP,
    RENDER_MIDA,

    RENDER_Count, // Do not use
};

extern const char* const renderingTypeNames[RENDER_Count];