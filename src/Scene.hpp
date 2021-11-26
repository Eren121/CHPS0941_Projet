#ifndef __SCENE_HPP__
#define __SCENE_HPP__

#include <vector>
#include "triangleMesh.hpp"
#include "volume.hpp"

enum {MESH_TYPE, VOLUME_TYPE};

class Scene {
    public: 
    
    Scene();

    /*Ajout d'objets dans la scène*/

    /*Suppression d'objet dans la scène*/
    private : 
    std::vector<TriangleMesh*> m_meshs;
    std::vector<Volume*> m_volumes;
};

#endif