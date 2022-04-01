#include "CheckerboardTexture.h"
#include "helper_cuda.h"
#include <GL/glew.h>
#include <vector>

void CheckerboardTexture::load(uchar4 color1, uchar4 color2)
{
    if(m_textureID != 0) {
        // N'allouer la texture que si on a pas appelé load()
        glGenTextures(1, &m_textureID);
    }

    glBindTexture(GL_TEXTURE_2D, m_textureID);
    
    // Ici, on ne veut pas d'interpolation pour que le damier soit bien visible
    // Et on voit répeter (sans miroir) les textures pour répéter le pattern en damier
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    const unsigned int width = 2;
    const unsigned int height = 2;

    // En réalité, en OpenGL, par défaut le début du tableau est le pixel en bas à gauche de la texture
    // Donc les lignes du damier est inversé par rapport à comme il est écrit dans le code
    // Cela donne :
    // C2 C1
    // C1 C2
    // Mais ce n'est pas très important pour l'usage ici (pattern répété)
    const std::vector<uchar4> pixels = {
        color1, color2,
        color2, color1
    };

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

CheckerboardTexture::~CheckerboardTexture()
{
    glDeleteTextures(1, &m_textureID);
    m_textureID = 0;
}