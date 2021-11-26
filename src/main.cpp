#include <cstdlib>
#include <iostream>

#include "screenDisplay.h"

int main(int argc, char** argv){

    std::cout << "hello !!" << std::endl;
    ScreenDisplay w(680,420);
    w.run();


    return EXIT_SUCCESS;
}