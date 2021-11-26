#ifndef CAMERA_H
#define CAMERA_H

#include "vec.h"

/**
    \brief Une structure qui permet de gérer la caméra Optix
    Ses valeurs sont modifiées en fonction de la caméra OpenGL
    Il s'agit simplement d'une structure permettant la jointure entre la caméra Optix et OpenGL
*/

#ifndef DEVICE
#endif
struct Cam {
     vec3f pos, at, up;
     float znear, zfar;
     float offset;
     float hfov, vfov;
#ifndef DEVICE
/*     void print() const{
         qDebug() << "cam position : " << pos.x << " " << pos.y << " " << pos.z;
         qDebug() << "cam at : " << at.x << " " << at.y << " " << at.z;
         qDebug() << "cam up : " << up.x << " " << up.y << " " << up.z;
     }
*/
     /**
        \brief Fonction qui convertie une structure Cam en QJsonObject
    */
  /*   QJsonObject toJson(){
        QJsonObject json;
        json.insert("positionX",pos.x);
        json.insert("positionY",pos.y);
        json.insert("positionZ",pos.z);
        json.insert("atX",at.x);
        json.insert("atY",at.y);
        json.insert("atZ",at.z);
        json.insert("upX",up.x);
        json.insert("upY",up.y);
        json.insert("upZ",up.z);
        json.insert("offset",offset);
        json.insert("hfov",hfov);
        json.insert("vfov",vfov);
        return json;
     }
*/
     /**
        \brief Initialise une structure Cam par un json
    */
  /*   void fromJson(QJsonObject json){
         pos    = vec3f(json.value("positionX").toDouble(),json.value("positionY").toDouble(),json.value("positionZ").toDouble());
         at     = vec3f(json.value("atX").toDouble(),json.value("atY").toDouble(), json.value("atZ").toDouble());
         up     = vec3f(json.value("upX").toDouble(),json.value("upY").toDouble(),json.value("upZ").toDouble());
         hfov   = json.value("hfov").toDouble();
         vfov   = json.value("vfov").toDouble();
         offset = json.value("offset").toDouble();
     }
*/
#endif
};

#endif // CAMERA_H
