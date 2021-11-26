#ifndef OPTIXVIEWER_H
#define OPTIXVIEWER_H

#include <QFile> //to read ptx file
#include <QDebug>
#include "camera.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "vec.h"
#include <optix.h>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_gl_interop.h>
#include <QFile>
#include <sstream>

#include "3dobject/volume.h"
#include "scene.h"
#include "QTransferFunction.h"
#include <qglviewer.h>

/**
    OptixRender est notre ray tracer. Cette objet nous permet de créer un context Optix,
    de l'initialiser et d'y charger les différentes informations de nos objets.

    Il est composé de nombreuses fonctions. Cependant, il gère seulement les échangent de nos données entre notre scèene et notre contexte Optix.
    Notre contexte optix est composé de différentes parties :
        -VAS : Volume Acceleration Structures
        -MAS : Mesh Acceleration Strcuture
        -IAS : Instance Acceleration Structure
        -Différent Module.
        -SBT : Shading Binding Table.
    Le VAS nous permet de gérer l'ensemble des informations qui non pas de maillage. Il gère donc l'ensemble des formes implicites et nos volumes.
    Le MAS nous permet de gérer les maillages triangulaires.
    L'IAS nous permet de gérer les intersections des VAS et MAS en un seul lancé de rayon. Il représente la "fusion" du VAS et MAS.
    Les modules représentent l'ensemble des .cu. Les .cu sont constituées de différent programmes :
    (Closest Hit program, Any Hit program, InterSection program). Ces programmes nous permettent de gérer le rendu et l'ensemble des itnersections.

    Le SBT nous permet de gérer les données propres aux objets de notre scènes, par exemple, le centre de notre volume.
*/

class OptixRender
{
public:
    OptixRender();
    OptixRender(Scene *modele);
    OptixRender(Scene *modele, const int nbGPU, const int numGPU);

    ~OptixRender();
    LaunchParams* getLaunchParams(){return &launchParams;}
    /**
        \brief Initialize le contexte OptiX, charge les différents modules pour réaliser les rendus.
        Créer les VAS, MAS et IAS pour pouvoir identifier les intersections et instancie la SBT.
    */
    void initialize(Scene *modele);

    /**
        \brief Réalise le rendu d'une image
    */
    void render();

    /**
        Dimensionne la taille de la fenêtre de rendu
    */
    void resize(const vec2i &newSize);

    /**
        \brief Permet de récupérer les données de l'image qui a été rendu
    */
    void downloadPixels(uint32_t h_pixels[]);

    void downloadPixels(uint32_t h_pixels[], size_t offset,size_t y_offset, size_t size);

    void setCamera(qglviewer::Camera* c0, const int index);
    void setCamera(const Cam &cam, const int index);
    /**
        Permet de fixer la cameré OptiX.
        La cameré passé en paramètre à les propriétés de la caméra OpenGL pour avoir une superposition correcte.
    */

    /* Ensemble de setter qui permette de changer les options de rendu durant l'execution*/
    void setRenderingMethode(unsigned char methodes);
    void setSampler(float value);
    void setLut(unsigned char lut);
    void setLutFunction(QTransferFunction* lutFunction);
    void setRange(float min, float max);
    void setMinIntensity(float min);
    void setMaxIntensity(float max);
    void setRepere(unsigned char value);
    void setBackground(unsigned char value);
    void setBoundingBox(unsigned char value);
    void setPlane(unsigned char value);
    void setRedLightColor(unsigned char value);
    void setGreenLightColor(unsigned char value);
    void setBlueLightColor(unsigned char value);
    void setLightPositionWithCamera(unsigned char v);
    void setLightPosX(float p);
    void setLightPosY(float p);
    void setLightPosZ(float p);
    void setDeplacementMode(bool b) { launchParams.frame.deplacementMode = b ? 1 : 0;}
    float getSampler() const;
    QTransferFunction* getLut() const;

    void loadVolume(QString path);
    int loadMesh(const QString &mesh);

    void updateSBTBuffer();

    Scene* getModele();
    QString getName(const int &id);
    void translateVolume(vec3f t);

    void buildIAS(Scene *modele, OptixTraversableHandle volume_traversable, OptixTraversableHandle mesh_traversable);
    void loadLut(const QString &path);
    /* helper function that initializes optix and checks for errors */
    void initOptix();

    /* creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /* creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModules();
    void createCylinderModule();
    void createVolumeModule();
    void createMeshModule();
    void createSphereModule();
    void createConeModule();
    void createBboxModule();
    /* does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();

    /* does all setup for the miss program(s) we are going to use */
    void createMissPrograms();

    /* does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();
    void createVolumeHitgroupPrograms();
    void createSphereHitgroupPrograms();
    void createMeshHithtoupPrograms();
    void createCylinderHitgroupPrograms();
    void createConeHithtoupPrograms();
    void createBboxHitgroupPrograms();
    /* assembles the full pipeline of all programs */
    void createPipeline();

    /* constructs the shader binding table */
    void buildSBT();

    /* build an acceleration structure for the given triangle mesh */
    void buildVolumeAccel();
    void buildMeshAccel();
    void buildBboxAccel();

    void updateAccelerationStructure();

    void updateSBT();
    //Mets à jours les données concernants les Meshs
    void updateMAS();
    //Mets à jours les données concernants les aabb structures
    void updateVAS();

    void updateIAS();

    /*Permet de demander la mise à jours du VAS,MAS  IAS ou de la SBT
        Ceci, permet d'éviter de régénérer une instance au besoin car la génération des  GAS est lourdes.
    */
    void notifyMeshChanges();
    void notifySbtChanges();
    void notifyAabbObjectChanges();

    void render(const int widthOffset, const int heightOffset, int nbXRayon, int nbYRayon);


    void getDepthMap(float *depthMap, const size_t size);
    void getObjectMap(int *depthMap, const size_t size);

    static void colorizeDepthMap(float *depthMap, float *colorizeDepthMap, const size_t size);
private :

    /* SBT record for a raygen program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here
      void *data;
    };

    // SBT record for a miss program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here
      void* data;
    };

    /* SBT record for a hitgroup program */
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here
      sbtData sbt = {};
    };
    struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupMeshRecord
    {
      __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
      // just a dummy value - later examples will use more interesting
      // data here

    };
  protected:
    /* @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /* @} */

    // the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /* @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};
    /* @} */

    /* @{ the module that contains out device programs */
    OptixModule                 volume_module;
    OptixModule                 cylinder_module;
    OptixModule                 mesh_module;
    OptixModule                 sphere_module;
    OptixModule                 cone_module;
    OptixModule                 raygen_module;
    OptixModule                 bbox_module;

    OptixModuleCompileOptions   moduleCompileOptions = {};
    /* @} */

    /* vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};
    CUDABuffer hitgroupMeshRecordsBuffer;
    OptixShaderBindingTable sbtMesh = {};

    /* @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;
    /* @} */

    CUDABuffer colorBuffer;
    CUDABuffer depthMapBuffer;
    CUDABuffer objectMapBuffer;

    CUDABuffer iasBuffer, masBuffer, vasBuffer/*, bboxasBuffer*/;
    OptixTraversableHandle mas{0};
    OptixTraversableHandle vas{0};
    OptixTraversableHandle ias{0};
//    OptixTraversableHandle bboxas{0};

    Scene* modele;
    bool isVolumeDataModified = false;
    bool isMeshDataModified = false;
    bool isSBTDataModified = false;

    QString ptx_volume_path = "./devicePrograms.ptx";
    QString ptx_cylinder_path = "./cylinder.ptx";
    QString ptx_sphere_path = "./sphere.ptx";
    QString ptx_mesh_path = "./mesh.ptx";
    QString ptx_cone_path = "./cone.ptx";
    QString ptx_raygen_path = "./raygen_multiGPU.ptx";
//    QString ptx_bbox_path = "./bbox.ptx";

    std::vector<HitgroupRecord> hitgroupRecords;


    //FOR VAS
    OptixAabb *aabb;
    CUdeviceptr d_aabb;
    CUDABuffer vasTempBuffer;
    CUDABuffer vasOutputBuffer;

    //FOR BBOXAS
//    OptixAabb *bbox;
//    CUdeviceptr d_bboxaabb;
//    CUDABuffer bboxasTempBuffer;
//    CUDABuffer bboxasOutputBuffer;
    //FOR MAS
    CUDABuffer masTempBuffer;
    CUDABuffer masOutputBuffer;

    //FOR IAS
    CUDABuffer iasTempBuffer;
    CUDABuffer iasOutputBuffer;


    QTransferFunction* qtf;
};

#endif // OPTIXVIEWER_H
