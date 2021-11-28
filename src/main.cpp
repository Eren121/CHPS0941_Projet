#include <cstdlib>
#include <iostream>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "screenDisplay.h"

int main(int argc, char** argv){

    std::cout << "hello !!" << std::endl;
    ScreenDisplay w(680,420);
    w.run();


    return EXIT_SUCCESS;
}