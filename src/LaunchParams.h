#ifndef LAUNCHPARAMS_H
#define LAUNCHPARAMS_H

#ifndef DEVICE
    #include <QJsonObject>
#endif
#include "optix7.h"
#include "CUDABuffer.h"
#include "vec.h"
#include "camera.h"
#include <stddef.h>
#define LP64
#define DEBUG 0
#define MIP 1
#define LMIP 2
#define DMIP 3
#define MIDA 4
#define DVR 5

#define BBOX 0
#define PLAN 1
#define SURFACE 2

#define REPERE 1


enum CAMERA_CONFIGURATION {PARALLELE_DECENTRE, PARALLELE, CONVERGENTE};
enum RENDER_TYPE {VOLUME_RENDERING=0, MESH_RENDERING,SPHERE_RENDERING,CYLINDER_RENDERING, CONE_RENDERING, BBOX_RENDERING};
enum STEREO {MONOSCOPIE=0, HOLO_STEREO, SEQUENTIAL_HOLO_INDEX, ANAGLYPH,SIDEbySIDE, QUADBUFFER};
//Cette structure permet de facilement faire la liaison binaire entre un float et un unsigned int.
//Elle est utilisé pour passer les données concernant les temps d'intersection sur le gpu
typedef struct time {
  union {
      float ftmin;
      unsigned int uitmin;
  }tmin;

  union {
      float ftmax;
      unsigned int uitmax;
  }tmax;

}intersection_time;

typedef struct Slice{
    vec3f p1,p2,p3;
    vec3f size;
}Slice;

//Cette structure représente différentes informations de la scène.
/*
    Elle est composé de 3 parties :
        -Traversable : l'objet nécessaire à optix pour calculer les intersections
        -Camera : une définition de la caméra gérée par la caméra OpenGL.
        -Frame : différentes extra-informations pour le rendu
*/
struct LaunchParams
{
    struct {
        uint32_t* colorBuffer; //resultat du rendu
        int* objectMap; //Segmentation de l'image par objet
        float *depthMap; //L'image de profondeur
        vec2i     size; // taille de l'image
        unsigned char type = DMIP; // rename to renderType
        float sampler = 0.001f;
        unsigned char enableLut = 0;
        float minIntensity = 0.0f, maxIntensity = 1.0f;
        vec4f *lut;
        unsigned char plane = 1; // change for boolean
        unsigned char bbox = 1; // change for boolean
        unsigned char background = 0; // change for boolean
        unsigned char repere = 1; // change for boolean
        vec3f lightColor = vec3f(1.0f);
        vec3f lightPosition;
        unsigned char lightOnCamera = 1;
        unsigned char stereo = MONOSCOPIE; // find other name
        unsigned char deplacementMode = 0; // change for boolean
        float moveReduction = 1.f;
        int modeCamera = PARALLELE_DECENTRE;
    } frame;

    struct {
        vec3f position[45];
        vec3f direction[45];
        vec3f horizontal[45];
        vec3f vertical[45];
        vec3f up[45];
        float offset[45];
        float cosFovY;
        float hfov,vfov;
        float znear,zfar;
    } camera;
    struct {
        float pitch = 1035.07f;
        float tilt= -0.0728167f;
        float center = -0.352053f;
        float subp = 4.340277882874943e-05;
        float pix = subp * 3.0f;

    }HoloParametre;

    struct {
        int nbGpu;
        int numGPU;
        int widthOffset;
        int heightOffset;
    } multi_gpu;

    OptixTraversableHandle traversable;

#ifndef DEVICE
    void fillJson(QJsonObject *json){
        json->insert("sizeX",         frame.size.x);
        json->insert("sizeY",         frame.size.y);
        json->insert("renderType",    frame.type);
        json->insert("sampler",       frame.sampler);
        json->insert("enableLut",     frame.enableLut);
        json->insert("minIntensity",  frame.minIntensity);
        json->insert("maxIntensity",  frame.maxIntensity);
        json->insert("enablePlane",   frame.plane);
        json->insert("bbox",          frame.bbox);
        json->insert("background",    frame.background);
        json->insert("repere",        frame.repere);
        json->insert("lightColorX",  frame.lightColor.x);
        json->insert("lightColorY",  frame.lightColor.y);
        json->insert("lightColorZ",  frame.lightColor.z);
        json->insert("lightPositionX",frame.lightPosition.x);
        json->insert("lightPositionY",frame.lightPosition.y);
        json->insert("lightPositionZ",frame.lightPosition.z);
        json->insert("lightOnCamera", frame.lightOnCamera);
        json->insert("stereo",        frame.stereo);
        json->insert("moveReduction", frame.moveReduction);
        json->insert("znear",         camera.znear);
        json->insert("zfar",          camera.zfar);
        json->insert("pitch",         HoloParametre.pitch);
        json->insert("tilt",          HoloParametre.tilt);
        json->insert("center",        HoloParametre.center);
        json->insert("subp",          HoloParametre.subp);
        json->insert("pix",           HoloParametre.pix);

    }

    void setFromJson(QJsonObject *json){
        frame.size.x          = json->value("sizeX").toInt();
        frame.size.y          = json->value("sizeY").toInt();
        frame.type            = json->value("renderType").toInt();
        frame.sampler         = json->value("sampler").toDouble();
        frame.enableLut       = json->value("enableLut").toInt();
        frame.minIntensity    = json->value("minIntensity").toDouble();
        frame.maxIntensity    = json->value("maxIntensity").toDouble();
        frame.plane           = json->value("enablePlane").toInt();
        frame.bbox            = json->value("bbox").toInt();
        frame.background      = json->value("background").toInt();
        frame.repere          = json->value("repere").toInt();
        frame.lightColor.x    = json->value("lightColorX").toDouble();
        frame.lightColor.y    = json->value("lightColorY").toDouble();
        frame.lightColor.z    = json->value("lightColorZ").toDouble();
        frame.lightPosition.x = json->value("lightPositionX").toDouble();
        frame.lightPosition.y = json->value("lightPositionY").toDouble();
        frame.lightPosition.z = json->value("lightPositionZ").toDouble();
        frame.stereo          = json->value("stereo").toInt();
        frame.moveReduction   = json->value("moveReduction").toDouble();
        camera.znear          = json->value("znear").toDouble();
        camera.zfar           = json->value("zfar").toDouble();
    }
#endif
};
/*
    La structure VolumetricCube permet de faire la liaison entre l'Objet Volume utilisé sur le host et
    et ses informations sur le device.
*/
struct VolumetricCube {
     vec3f size;
     vec3f center;
     vec3i sizePixel;
     cudaTextureObject_t tex = 0;
     int nbSlice = 0;
     Slice *slices;
     unsigned char isVisible = 1;
 };
/*
    La structure TriangleMeshSBT permet de faire la liaison entre l'Objet TriangleMesh utilisé sur le host et
    et ses informations sur le device.
*/
struct TriangleMeshSBT {
    vec3f size;
    vec3f center;
    vec3f kd;
    vec3f *vertex;
    vec3i *indices;
    vec2f *texCoord;
    cudaTextureObject_t tex = 0;
    unsigned char hasTexture = 0;
    unsigned char type = BBOX;
    unsigned char isVisible = 1;
 };

/*
    La structure SphereSBT permet de regrouper l'ensemble des informations pour afficher une sphère sur le device.
*/
typedef struct SphereSbt{
   vec3f position = vec3f(1.0f);
   vec3f color;
   float r = 1.0f;
   unsigned char isVisible =1;
}SphereSbt;


/*
    La structure CylinderSbt permet de regrouper l'ensemble des informations pour afficher un cylindre sur le device.
*/
typedef struct CylinderSbt{
   vec3f axe;
   vec3f color;
   vec3f position;
   float r = 1.0f;
   float h = 1.0f;
   unsigned char type = 0;
   unsigned char isVisible = 1;
}CylinderSbt;

/*
    La structure ConeSbt permet de regrouper l'ensemble des informations pour afficher un cone sur le device.
*/
typedef struct ConeSbt{
    vec3f position;
    vec3f color;
    vec3f v;
    float angle;
    float h;
    unsigned char type = 0;
    unsigned char isVisible = 1;
}ConeSbt;

/*
    La structure sbtData permet de regrouper l'ensemble des informations pour afficher
    l'ensemble des éléments affichables dans notre scène.
    Il est constitué de l'ensemble des structures précédentes
*/
typedef struct sbtData {

    //union{
        VolumetricCube volumeData;
        TriangleMeshSBT meshData;
        SphereSbt sphereSbt;
        CylinderSbt cylinderSbt;
        ConeSbt coneSbt;
   // };
        int id = -1;
}sbtData;



#if DEVICE==1
    /*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;
__device__ void setSegmentedAndDepthMap(const int id, const float tmin){
    const int offset = optixLaunchParams.frame.deplacementMode == 0 ? 1 : optixLaunchParams.frame.moveReduction;
    const vec3f rd = optixGetWorldRayDirection();
    const vec3f ro = optixGetWorldRayOrigin();
   //Calcul profondeur
    const float d = norme(rd * tmin);
   // const float d = 255.f;
    if( optixLaunchParams.frame.stereo == HOLO_STEREO){
        const int ix = optixLaunchParams.multi_gpu.widthOffset + (optixGetLaunchIndex().x * offset/3);
        const int iy = optixLaunchParams.multi_gpu.heightOffset + optixGetLaunchIndex().y * offset;
        const int i  = (optixGetLaunchIndex().x * offset)%3;
        optixLaunchParams.frame.depthMap[iy * optixLaunchParams.frame.size.x*3 + ix*3 +i] = d;
//        optixLaunchParams.frame.objectMap[iy * optixLaunchParams.frame.size.x * 3 + i + ix*3] = id;
    }
    else {
        const int ix = optixLaunchParams.multi_gpu.widthOffset +  optixGetLaunchIndex().x*(optixLaunchParams.frame.deplacementMode == 0 ? 1 : optixLaunchParams.frame.moveReduction);
        const int iy = optixLaunchParams.multi_gpu.heightOffset + optixGetLaunchIndex().y*(optixLaunchParams.frame.deplacementMode == 0 ? 1 : optixLaunchParams.frame.moveReduction);

        for(int i = 0; i < 3 ; ++i){
            optixLaunchParams.frame.depthMap[iy * optixLaunchParams.frame.size.x * 3 + i + ix*3] = d;
//            optixLaunchParams.frame.objectMap[iy * optixLaunchParams.frame.size.x * 3 + i + ix*3] = id;
        }
    }
}
#endif
#endif
