#ifndef CAMERA_H
#define CAMERA_H

#include "vec.h"

struct Camera {
    struct Camera(vec3f &newPos = vec3f(0.f,0.f,-5.f), vec3f &newAt = vec3f(0.f,0.f,0.f), vec3f &newUp = vec3f(0.f,1.f,0.f)) : pos(newPos),at(newAt), up(newUp){}

    vec3f getAt() const       {return at ;}
    vec3f getUp() const       {return up ;}
    vec3f getPosition() const {return pos;}

    vec3f at, up,  pos;    
};

#endif