#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <vector>
#include "triangleMesh.h"
//#include "volume.hpp"

class Scene {
    public: 
    Scene();

    /*Ajout d'objets dans la scène*/
    void addMesh(TriangleMesh* mesh);
    /*Suppression d'objet dans la scène*/
    void rmMesh(const int id);

    size_t getNumVolume() const;
    size_t getNumMesh() const;
    private : 
    std::vector<TriangleMesh*> m_meshs;
   // std::vector<Volume*> m_volumes;
};

#endif