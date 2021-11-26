#include "Scene.hpp"

Scene::Scene(){}
/*Ajout d'objets dans la scène*/
void Scene::addMesh(TriangleMesh* mesh){
    m_meshs.push_back(mesh);
}
/*Suppression d'objet dans la scène*/
void Scene::rmMesh(const int id){
    m_meshs.erase(m_meshs.begin() + id);
}


size_t Scene::getNumVolume() const{
    return 0;
}

size_t Scene::getNumMesh() const{
    return m_meshs.size();
}