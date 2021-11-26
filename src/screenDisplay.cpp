#include "screenDisplay.h"

void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

ScreenDisplay::ScreenDisplay(const int width, const int height, const std::string title) : m_screenSize(width,height),m_windowTitle(title){

    if (!glfwInit())
    {
        // Initialization failed
        std::cerr << "GLFW initialization failed..." << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    
    window = glfwCreateWindow(m_screenSize.x, m_screenSize.y, m_windowTitle.c_str(), NULL, NULL);
    if (!window)
    {
        // Window or OpenGL context creation failed
        std::cerr << "GLFW window's creation failed..." << std::endl;
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, key_callback);

    pixels.resize(width*height);

    //Creation des objets de la scene
    createSceneEntities();
}


ScreenDisplay::~ScreenDisplay(){
    glfwDestroyWindow(window);
    glfwTerminate();

}


void ScreenDisplay::createSceneEntities(){
    //Creation de la scene a afficher
    //Ajout d'un cube unitaire dans la scene
    TriangleMesh *t= new TriangleMesh();
    t->addUnitCube();
    scene.addMesh(t);
}

void ScreenDisplay::run(){
    std::cout << "ScreenDisplay::run()";
    while (!glfwWindowShouldClose(window))
    {
        // Keep running
        float ratio;
        int width, height;
 
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
 
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glClearColor(1.f,0.f,0.f,1.f);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void ScreenDisplay::resize(const int width, const int height){
    m_screenSize.x = width;
    m_screenSize.y = height;
}

vec2i ScreenDisplay::getSize() const {
    return m_screenSize;
}
int   ScreenDisplay::getWidth()  const {
    return m_screenSize.x;
}
int   ScreenDisplay::getHeight() const {
    return m_screenSize.y;
}

