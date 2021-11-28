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

    //Création du render
    optixRender = new OptixRender(&scene);
    optixRender->resize(m_screenSize);
    
}


ScreenDisplay::~ScreenDisplay(){
    delete(optixRender);
    glfwDestroyWindow(window);
    glfwTerminate();
}


void ScreenDisplay::createSceneEntities(){
    //Creation de la scene a afficher
    //Ajout d'un cube unitaire dans la scene
    TriangleMesh *t= new TriangleMesh();
    TriangleMesh *t2= new TriangleMesh();
    t->addUnitCube();
    t2->addUnitCube();
    t2->translate(vec3f(3.f,0.f,0.f));
    t2->setColor(vec3f(0.f,1.f,0.f));
   // scene.addMesh(t);
    scene.addMesh(t2);
    Volume *v = new Volume();
    scene.addVolume(v);
    

}

void ScreenDisplay::run(){
    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float) height;
       
        update();
        render();
        drawScene();
        // Keep running
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}


void ScreenDisplay::update(){
    optixRender->setCamera(m_camera);
}
void ScreenDisplay::render(){
    optixRender->render();
    optixRender->downloadPixels(pixels.data());
}

void ScreenDisplay::drawScene(){
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.f,0.f,0.f,1.f);
    if (fbTexture == 0)
        glGenTextures(1, &fbTexture);


    glBindTexture(GL_TEXTURE_2D, fbTexture);

    // set basic parameters
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_screenSize.x, m_screenSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, m_screenSize.x, m_screenSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)m_screenSize.x, 0.f, (float)m_screenSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.f, 0.f);
        glVertex3f(0.f, 0.f, 0.f);
        glTexCoord2f(0.f, 1.f);
        glVertex3f(0.f, (float)m_screenSize.y, 0.f);

        glTexCoord2f(1.f, 1.f);
        glVertex3f((float)m_screenSize.x, (float)m_screenSize.y, 0.f);

        glTexCoord2f(1.f, 0.f);
        glVertex3f((float)m_screenSize.x, 0.f, 0.f);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);
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

