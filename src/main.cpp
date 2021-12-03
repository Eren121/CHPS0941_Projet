#include <cstdlib>
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "screenDisplay.h"

bool ScreenDisplay::translation;
bool ScreenDisplay::rotation;
vec3f ScreenDisplay::translateCamera;
vec2f ScreenDisplay::oldCursorPosition;
Camera ScreenDisplay::m_camera;
vec2f ScreenDisplay::coordonneeSpherique;


int main(int argc, char** argv){

    std::cout << "hello !!" << std::endl;
    ScreenDisplay w(680,420);
    w.run();


    return EXIT_SUCCESS;
}