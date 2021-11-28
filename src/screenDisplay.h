#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

#include "vec.h"
#include "camera.h"
#include "Scene.hpp"
#include "optixRender.h"

class ScreenDisplay {
    public :

        ScreenDisplay(const int width = 680, const int height = 420,const std::string title = "VSProject");
        ~ScreenDisplay();

        void createSceneEntities();

        void run();
        
        void update();
        void render();
        void drawScene();

        void resize(const int width, const int height);

        vec2i getSize() const;
        int getWidth()  const;
        int getHeight() const;

    private :
    
    vec2i m_screenSize;
    GLFWwindow* window;
    std::string m_windowTitle = "VSProject" ;
    Camera m_camera;
    std::vector<uint32_t> pixels;
    GLuint                fbTexture {0};
    Scene scene;
    OptixRender *optixRender;
};