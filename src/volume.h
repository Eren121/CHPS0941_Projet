#ifndef VOLUME_H
#define VOLUME_H

#include "vec.h"
#include "CUDABuffer.h"
#include <QString>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_texture_types.h>
#include <cuda_runtime_api.h>
#include <texture_fetch_functions.h>
#include <optix_types.h>
#include <QDebug>
#include "LaunchParams.h"
#include "aabbobject.h"


#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"


/**
    \class Volume
    \brief Represente un Volume
    Un volume est définis par une position dans l'espace, une dimension dans l'espace, une dimension pour les données
    et un ensemble de donnée qui traduis les valeurs des voxels.
    Pour stocker les données, nous utilisons des textures ce qui nous permettent une interpolation tri-linéaire à la récupération des valeurs
    et de meilleurs performances.
*/
class Volume : public AabbObject
{
public:
    Volume();
    Volume(QString path);
    ~Volume();

    /**
        \brief Fonction tempons pour appeler la bonne fonction de chargement de volume.
    */
    void loadVolume(const QString &path);

    /**
        \brief Charge un volume sous le format DAT
    */
    void loadDATVolume(const QString &path);

    /**
        \brief Charge un volume sous le format B8
    */
    void loadB8Volume(const QString &path);

    /**
        \brief Charge un volume sous le format Dicom grace a ITK
    */
    void loadDicomVolume(const QString &path);

    /**
       \Brief Créer la texture à partir des données chargées.
    */
    void createTexture();

    /**
        \brief Realise la rotation du volume ----> pas implemente
    */
    virtual void rotate(const float &rx, const float &ry, const float &rz) override ;

    /**
        \brief Realise la rotation du volume ----> pas implemente
    */
    virtual void rotate(const QQuaternion &q) override ;


    /**
        \brief Redimensioonne le volume par un coefficient multiplicateur
    */
    virtual void scale(const vec3f &d) override ;

    /**
        \brief Set la taille du volume par la valeur newDIm
    */
    virtual void resize(const vec3f &newDim) override ;

    /**
       \brief Retourne la dimension dans l'espace
    */
    vec3f getWorldSize() const;

    /**
       \brief Retourne la position dans l'espace
    */
    vec3f getWorldCenter() const;

    /**
       \brief Retourne la dimension en voxel
    */
    vec3i getPixelSize() const;

    /**
        \brief Retourne un tableau representant les valeurs de la texture brut ( soit de type CHAR ou SHORT)
    */
    void* getData() const { return pData;}

    /**
        \brief Retourne le type de donnees lus des fichiers : SHORT ou CHAR
    */
    int getDataType() const {return type;}
    /**
       \brief Retourne le pointeur de la texture 3D représentant les données
    */
    cudaArray_t getTextureDevicePointer(const int iGPU = 0 ) const;

    /**
       \brief Retourne la texture 3d représentant les données
    */
    cudaTextureObject_t getTextureReference(const int iGPU = 0) const;

    /**
        \brief REmplis results par les valeurs de la aabb du volume
    */
    virtual void getAabb( float results[6]) override;

    /**
        \brief Remplis la structure sbtData par les donnees utiles du volume pour le GPU = iGPU
    */
    virtual void getSbt(sbtData *sbt, const int iGPU = 0) override;

    /**
        \brief Ajoute un plan de decoupe sur le volume
    */
    void addSlice(Slice slice);

    /**
        \brief Ajoute un plan de decoupe definis par 3 points
    */
    void addSlice(vec3f p1, vec3f p2, vec3f p3);

    /**
        \brief Supprime tout les plans de decoupes du volumes
    */
    bool clearSlice();

    /**
        \brief Supprime la slice dont l'indice est passe en parametre
    */
    void rmSlice(const int id);

    /**
        \brief Converti un volume en un QJsonObject
    */
    virtual QJsonObject toJson() override;

private :
    enum DATA_TYPE{SHORT,UCHAR}; //type de donnees utilisees par le fichier pour definir pData
    void *pData;  //data representant la texture "brute"
    unsigned char type;//type de donnees utilisees par le fichier pour definir pData
    vec3f worldSize; //La taille du volume
    vec3i pixelSize; //la dimension du volume en voxel
    unsigned short maxIntensity = 0; //l'intensite max de la texture du volume
    unsigned short minIntensity = 255; //l'intensite min de la texture

    cudaTextureObject_t *cudaTexture; //Texture par GPU
    cudaArray_t *d_array; //pointeur vers le tableau de la texture par GPU

    //slice
    std::vector<Slice> slices; //Listes de slices
    CUDABuffer *sliceBuffer; //Listes de slice sur le gpu

};

#endif // VOLUME_H
