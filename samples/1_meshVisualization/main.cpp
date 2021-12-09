#define MESH_VISUALIZATION 1
#include <cstdlib>
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "screenDisplay.h"
#include "LaunchParams.h"

bool ScreenDisplay::translation;
bool ScreenDisplay::rotation;
vec3f ScreenDisplay::translateCamera;
vec2f ScreenDisplay::oldCursorPosition;
Camera ScreenDisplay::m_camera;
vec2f ScreenDisplay::coordonneeSpherique;
vec2f ScreenDisplay::ihmpos; 
vec2f ScreenDisplay::ihmsize;


int main(int argc, char** argv){
    std::cout << "LaunchParams :" << sizeof(LaunchParams) << std::endl;
    ScreenDisplay w(680,420);
    w.run();


    return EXIT_SUCCESS;
}