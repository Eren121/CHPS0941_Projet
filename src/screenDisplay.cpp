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

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
        std::cout << "mouse right button pressed" << std::endl;
        ScreenDisplay::translation = true;        
    }
    else if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE){
        std::cout << "mouse right button released" << std::endl;
        ScreenDisplay::translation = false; 
    }
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
        std::cout << "mouse left button pressed" << std::endl;
        ScreenDisplay::rotation = true;        
    }
    else if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
        std::cout << "mouse left button released" << std::endl;
        ScreenDisplay::rotation = false; 
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if( ScreenDisplay::rotation){

    }
    if(ScreenDisplay::translation){
        vec2f t = ScreenDisplay::oldCursorPosition - vec2f(xpos,ypos);
        ScreenDisplay::m_camera.pos.x += t.x*0.005;
        ScreenDisplay::m_camera.pos.y += t.y*0.005; 
        ScreenDisplay::m_camera.at.x  += t.x*0.005;
        ScreenDisplay::m_camera.at.y  += t.y*0.005; 
    }
    ScreenDisplay::oldCursorPosition = vec2f(xpos,ypos);
}

ScreenDisplay::ScreenDisplay(const int width, const int height, const std::string title) : m_screenSize(width,height),m_windowTitle(title){

    if (!glfwInit())
    {
        // Initialization failed
        std::cerr << "GLFW initialization failed..." << std::endl;
        exit(EXIT_FAILURE);
    }

    glfwSetErrorCallback(error_callback);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
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
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    //Initialisation of Imgui
   // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.DisplaySize.x = width;io.DisplaySize.y = height;
    io.Fonts->Build();
   
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    const char* glsl_version = "#version 130";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


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
    scene.addMesh(t2);
    Volume *v = new Volume();
    scene.addVolume(v);
    

}

void ScreenDisplay::run(){
    while (!glfwWindowShouldClose(window))
    {
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        updateInterface();
        update();
        render();
        drawScene();
        
        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //keep running
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void ScreenDisplay::updateInterface(){
    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    ImGui::Begin("Scene controler");
    
    if(ImGui::Button("Load volume")){ 
        std::string volume_path;
        std::cout << "Enter path to load volume" << std::endl;
        std::cin >> volume_path;
        Volume *v = new Volume(volume_path);
        scene.addVolume(v);
        optixRender->notifyAabbObjectChanges();
        optixRender->notifySbtChanges();
    }
    if(ImGui::Button("clean scene")){
        scene.clear();
        optixRender->notifyMeshChanges();
        optixRender->notifyAabbObjectChanges();
        optixRender->notifySbtChanges();
    }
    if(ImGui::Button("add mesh")){
        TriangleMesh *m = new TriangleMesh();
        m->addUnitCube();
        scene.addMesh(m);
        optixRender->notifyMeshChanges();
        optixRender->notifySbtChanges();
    }
    LaunchParams *parameters = optixRender->getLaunchParams();
    ImGui::SliderFloat("Number of sample", &parameters->frame.sampler, 1.f, 9999.f);
    ImGui::SliderFloat("Min intensity", &parameters->frame.minIntensity, 0.f, 1.f);
    ImGui::SliderFloat("Max Intensity", &parameters->frame.maxIntensity, 1.f, 0.f);
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

