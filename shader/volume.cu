// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define DEVICE 1
#define nbSamples optixLaunchParams.frame.sampler

#include <optix_device.h>

#include "../common/LaunchParams.h"
#include "../render/ip.h"

  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }
  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------

  __device__ void swap(float &a, float &b) {
    float tmp = a;
    a = b;
    b = tmp;
  }
  __device__ bool inVolume(const VolumetricCube &data, const vec3f &pi){
        bool res = false;
        const vec3f min = data.center - data.size/2.0f;
        const vec3f max = data.center + data.size/2.0f;
        if(( pi.x <= max.x && pi.x >= min.x)&&
                ( pi.y <= max.y && pi.y >= min.y) &&
                ( pi.z <= max.z && pi.z >= min.z))
            res = true;
        return res;
  }
  
/**
 * La méthode intensityProjection() accepte
 * une classe générique en argument qui permet d'implémenter une méthode d'IP
 * générique sans dupliquer le code.
 * 
 * Cette classe itère toutes les intensités de voxels rencontrés (du plus près au plus loin)
 * et retourne la valeur
 * correspondant à la méthode choisie suivant les voxels rencontrés.
 * 
 * On n'utilise pas les méthodes virtuelles (une autre façon de faire qui aurait été possible)
 * car c'est délicat à utiliser avec CUDA quand on passe un pointeur d'une classe host -> device.
 */
  template<typename IntensityProjection>
  __device__ void intensityProjection(IntensityProjection& ip)
  {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const LutData& lut = optixLaunchParams.renderingTypeOptions.lut;

      const VolumetricCube& data
      = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
    const int   primID = optixGetPrimitiveIndex();
    intersection_time time;
    time.tmin.uitmin = optixGetAttribute_0();
    time.tmax.uitmax = optixGetAttribute_1();
    //Gather information
    vec3f ro = optixGetWorldRayOrigin();
    vec3f rd = optixGetWorldRayDirection();
    vec3f& prd = *(vec3f*)getPRD<vec3f>();
    vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);

    //Ray
    vec3f point_in = ro + time.tmin.ftmin * rd ;
    vec3f point_out = ro + time.tmax.ftmax * rd;
    vec3f ray_world = point_out - point_in;

    // On est à l'intérieur de la bounding box
    // Pour DEMIP on stocke la profondeur pour pouvoir
    // calculer la profondeur localement à l'objet
    // On peut récupérer la profondeur grâce à ray_world / current_ray_length
    // Il semble que current_ray_length varie dans [0;1] 
    // et indique la profondeur dans la bounding box
    // current_ray_length = 1.0 => Le moins profond possible d'un point de vue de la caméra
    // current_ray_length = 0.0 => Le plus profond possible d'un point de vue de la caméra

    VoxelHitData hitData(optixLaunchParams.renderingTypeOptions);
    
    const float stepSize_current = norme(point_out - point_in) / nbSamples;
    vec3f step_vector_tex = normalize(ray_world) * stepSize_current;
    float current_ray_length = norme(ray_world);

    vec3f current_pos_tex = point_in;
    float current_max = 0.0f;
    float current_intensity = 0.0f;

    ip.setLut(lut);


    // Il est possible qu'il n'y ai aucun point de collision
    // Car on tire les rayons dans une AABB et donc aux bords cela peut être en dehors du modèle
    // Cela peut êter un cas particulier pour certaines méthodes comme AIP, où on doit diviser par le nombre
    // de voxels rencontrés (donc par zéro)
    // Pour éviter ce cas là,
    // On vérifie si au moins un voxel est rencontré, sinon on donne toujours une couleur par défaut.

    bool atLeastOneHit = false;

    while(current_ray_length > 0.0f){
      vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;
      current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);

      if(current_intensity != 0.0f) {
        // Quoi qu'il arrive, on considère qu'une intensité 0.0 signifie du vide
        // Géré à part pour éviter les cas où frame.minIntensity == 0.0 par ex.
        

        if( current_intensity >= optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity){
          atLeastOneHit = true;
          
          hitData.intensity = current_intensity;
          hitData.depth = 1.0f - current_ray_length;

          if(!ip.nextVoxelHit(hitData)) {
            break;
          }
        }
      }
      
      current_pos_tex = current_pos_tex + step_vector_tex;
      current_ray_length -= stepSize_current;
    }
    
    if(atLeastOneHit)
    {
      // On n'utilise pas le cannal alpha ici,
      // Le cannal alpha est seulement utilisé lors du traçage du rayon
      // pour certains algorithmes (MIDA)
      const float4 color = ip.getFinalColor();

      prd = vec3f(color.x, color.y, color.z);
      
      // Sauvegarde en sortie (utile pour DEMIP)
      ip.deepestDepthHit = hitData.depth;
    }
    else
    {
      prd = vec3f(0.0f); // Black
    }
  }

  __device__ void mip() {
      const VolumetricCube& data
       = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;
     const int   primID = optixGetPrimitiveIndex();
     intersection_time time;
     time.tmin.uitmin = optixGetAttribute_0();
     time.tmax.uitmax = optixGetAttribute_1();
     //Gather information
     vec3f ro = optixGetWorldRayOrigin();
     vec3f rd = optixGetWorldRayDirection();
     vec3f& prd = *(vec3f*)getPRD<vec3f>();
     vec3f sizeP = vec3f(data.sizePixel.x, data.sizePixel.y, data.sizePixel.z);

     //Ray
     vec3f point_in = ro + time.tmin.ftmin * rd ;
     vec3f point_out = ro + time.tmax.ftmax * rd;
     vec3f ray_world = point_out - point_in;

     const float stepSize_current = norme(point_out - point_in) / nbSamples;
     vec3f step_vector_tex = normalize(ray_world) * stepSize_current;
     float current_ray_length = norme(ray_world);

     vec3f current_pos_tex = point_in;
     float current_max = 0.0f;
     float current_intensity = 0.0f;

     //MIP
     prd = vec3f(0.0f);
     while(current_ray_length > 0.0f){
        vec3f pos_tex = (current_pos_tex - data.center + data.size / 2.0f) / data.size;
        current_intensity = tex3D<float>(data.tex,pos_tex.x,pos_tex.y,pos_tex.z);

        // On utilise > et pas >= car on considère qu'une intensité 0.0
        // est toujours vide (donc même si l'utilisateur met à intensity_min = 0.0 il n'entrera
        // pas dans la condition là ou il y a du vide)
        if( current_intensity > optixLaunchParams.frame.minIntensity && current_intensity <= optixLaunchParams.frame.maxIntensity){
            if( current_intensity > current_max )
                current_max = current_intensity;

            if( current_max == 1.0f){
                prd = vec3f(1.0f);
                break;
            }
            prd = vec3f(current_max);
        }
        current_pos_tex = current_pos_tex + step_vector_tex;
        current_ray_length -= stepSize_current;
     }
  }

  extern "C" __global__ void __closesthit__volume_radiance(){
      const VolumetricCube& data
       = (*(const sbtData*)optixGetSbtDataPointer()).volumeData;

      
      vec3f& prd = *(vec3f*)getPRD<vec3f>();
      RenderingTypeOptions& options = optixLaunchParams.renderingTypeOptions;

      switch(optixLaunchParams.renderingType)
      {
        case RENDER_MIP:
          {
            //mip();
            //équivalent

            MIP ip(options);
            intensityProjection(ip);
          }
          break;

        case RENDER_AIP:
          {
            AIP ip(options);
            intensityProjection(ip);
          }
          break;

        case RENDER_MINIP:
          {
            MinIP ip(options);
            intensityProjection(ip);
          }
          break;

        case RENDER_LMIP:
          {
            LMIP ip(options);
            intensityProjection(ip);
          }
          break;

        case RENDER_DEMIP:
          {
            if(options.demip.showDepthOnly)
            {
              DepthOnly ip(options);
              intensityProjection(ip);
            }
            else
            {
              // On doit ici tirer 2 rayons :
              // - le 1ier en MIP
              // - le 2ième en DEMIP qui utilise la couleur trouvée lors du 1ier rayon

              MIP ip1(options);
              intensityProjection(ip1);
              
              const float4 material = ip1.getFinalColor();
              
              DEMIP ip2(options, material, ip1.deepestDepthHit);
              intensityProjection(ip2);
            }
          }
          break;

        case RENDER_MIDA:
          {
            MIDA ip(options);
            intensityProjection(ip);
          }
          break;
      }
      
  }


  extern "C" __global__ void __anyhit__volume_radiance()
  {
  }


  extern "C" __global__ void __intersection__volume() {
      const VolumetricCube& sbtData
          = *(const VolumetricCube*)optixGetSbtDataPointer();
      vec3f ro = optixGetWorldRayOrigin();
      vec3f rayDir = optixGetWorldRayDirection();
      vec3f min, max;
      min = sbtData.center - sbtData.size / 2;
      max = sbtData.center + sbtData.size / 2;

      float tmin = (min.x - ro.x) / rayDir.x;
      float tmax = (max.x - ro.x) / rayDir.x;

      if (tmin > tmax) swap(tmin, tmax);

      float tymin = (min.y - ro.y) / rayDir.y;
      float tymax = (max.y - ro.y) / rayDir.y;

      if (tymin > tymax) swap(tymin, tymax);

      //Rayon en dehors du cube (normalement impossible )
      if ((tmin > tymax) || (tymin > tmax))

        return ;

      if (tymin > tmin)
          tmin = tymin;

      if (tymax < tmax)
          tmax = tymax;

      float tzmin = (min.z - ro.z) / rayDir.z;
      float tzmax = (max.z - ro.z) / rayDir.z;

      if (tzmin > tzmax) swap(tzmin, tzmax);

      //Rayon en dehors du cube (normalement impossible )
      if ((tmin > tzmax) || (tzmin > tmax))
        return ;

      if (tzmin > tmin)
          tmin = tzmin;

      if (tzmax < tmax)
          tmax = tzmax;
      if (tmin > tmax) swap(tmin, tmax);
      intersection_time time;

      time.tmin.ftmin = tmin;
      time.tmax.ftmax = tmax;
      optixReportIntersection(tmin, 1,time.tmin.uitmin, time.tmax.uitmax);
  }
