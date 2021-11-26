#include "3dobject/volume.h"

Volume::Volume() : AabbObject(vec3f(0.0f))
{
    worldSize = vec3f(1.0f);
    pixelSize = vec3i(0);
    minIntensity = 4096;
    maxIntensity = 0;
    renderType  = VOLUME_RENDERING;
    name = QString("Volume_%1").arg(id);

    sliceBuffer = new CUDABuffer[NB_OPTIX];
    for(int i = 0; i < NB_OPTIX; ++i){
       sliceBuffer[i].alloc(1);
    }

    cudaTexture = new cudaTextureObject_t[NB_OPTIX];
    d_array = new cudaArray_t[NB_OPTIX];
}

Volume::Volume(QString path){
    loadVolume(path);
    createTexture();
    renderType  = VOLUME_RENDERING;
    name = QString("Volume_%1").arg(id);
    for(int i = 0; i  < NB_OPTIX; ++i)
    	sliceBuffer[i].alloc(0);
    cudaTexture = new cudaTextureObject_t[NB_OPTIX];
    d_array = new cudaArray_t[NB_OPTIX];
}

Volume::~Volume(){
    free(pData);
    free(cudaTexture);
    cudaFree(d_array);
    for(int i = 0 ; i < NB_OPTIX; ++i)
    	sliceBuffer[i].free();
     delete[] sliceBuffer;
}

void Volume::rotate(const float &rx, const float &ry, const float &rz){}
void Volume::rotate(const QQuaternion &q){}

void Volume::scale(const vec3f &d){
    worldSize = worldSize + worldSize * d ;
}


void Volume::loadVolume(const QString &path){
    const QString ext = path.split('.').last();
    if(ext == "dat"){
        loadDATVolume(path);
    }else if(ext == "b8"){
        loadB8Volume(path);
    }
    if(!path.contains('.')){
        loadDicomVolume(path);
    }
}


void Volume::loadDATVolume(const QString &path){
    FILE* fp = fopen(path.toStdString().c_str(), "rb");
    unsigned short vuSize[3];
    fread((void*)vuSize, 3, sizeof(unsigned short), fp);
    pixelSize = vec3i(int(vuSize[0]),int(vuSize[1]),int(vuSize[2]));
    unsigned int uCount = int(vuSize[0]) * int(vuSize[1]) * int(vuSize[2]);
    unsigned short *data = new unsigned short[uCount];
    fread((void*)data, uCount, sizeof(unsigned short), fp);
    fclose(fp);
    for(unsigned int i = 0; i < uCount; ++i){
        if( data[i] > maxIntensity)
            maxIntensity = data[i];
        if( data[i] < minIntensity)
            minIntensity = data[i];
    }
    pData = data;
    type = SHORT;

    createTexture();

    /*Set Dimension*/
    vec3f size = worldSize;
    size = vec3f(size.x * (float)pixelSize.x,size.y * (float)pixelSize.y,size.z * (float)pixelSize.z );
    float max = size.x;
    if( max < size.y )
        max = size.y;
    if( max < size.z)
        max = size.z;
    worldSize = size / max * worldSize;
}

void Volume::loadB8Volume(const QString &path){
    FILE* fp = fopen(path.toStdString().c_str(), "rb");
    unsigned short vuSize[3];
    fread((void*)vuSize, 3, sizeof(unsigned short), fp);
    pixelSize = vec3i(int(vuSize[0]),int(vuSize[1]),int(vuSize[2]));

    unsigned int uCount = int(vuSize[0]) * int(vuSize[1]) * int(vuSize[2]);
    unsigned char *data = new unsigned char[uCount];
    fread((void*)data, uCount, sizeof(unsigned char), fp);

    fclose(fp);
    for(unsigned int i = 0; i < uCount; ++i){
        if( data[i] > maxIntensity)
            maxIntensity = data[i];
        if( data[i] < minIntensity)
            minIntensity = data[i];
    }
    pData = data;
    type = UCHAR;
    createTexture();

    /*Set Dimension*/
    vec3f size = worldSize;
    size = vec3f(size.x * (float)pixelSize.x,size.y * (float)pixelSize.y,size.z * (float)pixelSize.z );
    float max = size.x;
    if( max < size.y )
        max = size.y;
    if( max < size.z)
        max = size.z;
    worldSize = size / max * worldSize;
}


void Volume::loadDicomVolume(const QString &path){
    using PixelType = short;
    constexpr unsigned int Dimension = 3;

    using ImageType = itk::Image<PixelType, Dimension>;

    using ReaderType = itk::ImageSeriesReader<ImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    using ImageIOType = itk::GDCMImageIO;
    ImageIOType::Pointer dicomIO = ImageIOType::New();

    reader->SetImageIO(dicomIO);
    using NamesGeneratorType = itk::GDCMSeriesFileNames;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

    nameGenerator->SetUseSeriesDetails(true);
    nameGenerator->AddSeriesRestriction("0008|0021");

    nameGenerator->SetDirectory(path.toStdString().c_str());

    try
    {
      std::cout << std::endl << "The directory: " << std::endl;
      std::cout << std::endl << path.toStdString().c_str() << std::endl << std::endl;
      std::cout << "Contains the following DICOM Series: ";
      std::cout << std::endl << std::endl;
      using SeriesIdContainer = std::vector<std::string>;

      const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

      auto seriesItr = seriesUID.begin();
      auto seriesEnd = seriesUID.end();
      while (seriesItr != seriesEnd)
      {
        std::cout << seriesItr->c_str() << std::endl;
        ++seriesItr;
      }
      std::string seriesIdentifier;
      seriesIdentifier = seriesUID.begin()->c_str();


      std::cout << std::endl << std::endl;
      std::cout << "Now reading series: " << std::endl << std::endl;
      std::cout << seriesIdentifier << std::endl;
      std::cout << std::endl << std::endl;
      using FileNamesContainer = std::vector<std::string>;
      FileNamesContainer fileNames;

      fileNames = nameGenerator->GetFileNames(seriesIdentifier);
      reader->SetFileNames(fileNames);

      try
      {
        reader->Update();
      }
      catch (const itk::ExceptionObject & ex)
      {
        std::cout << ex << std::endl;
      }

      ImageType::Pointer image = reader->GetOutput();
      const ImageType::SizeType sizeOfImage = image->GetLargestPossibleRegion().GetSize();

      std::cout << " Size image:  " << sizeOfImage.at(0) << " " << sizeOfImage.at(1) << " " << sizeOfImage.at(2) << std::endl;
      pixelSize = vec3i(int(sizeOfImage.at(0)),int(sizeOfImage[1]),int(sizeOfImage[2]));
      unsigned int uCount = int(pixelSize.x) * int(pixelSize.y) * int(pixelSize.z);
       short *data = new  short[uCount];
       short minV = 9999;
       short maxV = 0;
      for(int x = 0; x < sizeOfImage[0]; ++x){
          for(int y = 0 ; y < sizeOfImage[1]; ++y){
              for(int z = 0; z < sizeOfImage[2]; ++z){
                  const ImageType::IndexType pixelIndex = {{ x,y,z }};
                  ImageType::PixelType pixelValue = image->GetPixel(pixelIndex);
                 if (minV > pixelValue)
                     minV = (float)pixelValue;
                 if( maxV < pixelValue)
                     maxV = (float)pixelValue;
              }
          }
      }
      for(int x = 0; x < sizeOfImage[0]; ++x){
          for(int y = 0 ; y < sizeOfImage[1]; ++y){
              for(int z = 0; z < sizeOfImage[2]; ++z){
                  const ImageType::IndexType pixelIndex = {{ x,y,z }};
                  ImageType::PixelType pixelValue = image->GetPixel(pixelIndex);
                  data[z * pixelSize.y * pixelSize.x + y * pixelSize.x + x] = maxV - pixelValue;
              }
          }
      }

      for(unsigned int i = 0; i < uCount; ++i){
          if( data[i] > maxIntensity)
              maxIntensity = data[i];
          if( data[i] < minIntensity)
              minIntensity = data[i];
      }
      pData = data;
      type = SHORT;
      createTexture();

      /*Set Dimension*/
      vec3f size = worldSize;
      size = vec3f(size.x * (float)pixelSize.x,size.y * (float)pixelSize.y,size.z * (float)pixelSize.z );
      float max = size.x;
      if( max < size.y )
          max = size.y;
      if( max < size.z)
          max = size.z;
     // worldSize = size / max * worldSize;

    }
    catch (const itk::ExceptionObject & ex)
    {
      std::cout << ex << std::endl;
    }
}


void Volume::createTexture(){
    float* h_array;

    unsigned int uCount = pixelSize.x * pixelSize.y * pixelSize.z;

    h_array = (float*)malloc(uCount * sizeof(float));

    for(unsigned int i = 0; i < uCount; ++i){
        switch(type){
            case SHORT :
                h_array[i] = (float)(((unsigned short*)(pData))[i]) / 4096.0f;
            break;
            case UCHAR:
                h_array[i] = (float)(((unsigned char*)(pData))[i]) / 255.0f;
            break;
        }
    }

    for(int i = 0; i < NB_OPTIX; ++i){
        cudaSetDevice(i);
        cudaChannelFormatDesc channel_descriptor = cudaCreateChannelDesc<float>();
        cudaExtent volumeSize = make_cudaExtent(pixelSize.x,pixelSize.y,pixelSize.z);
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc = {};

        cudaMalloc3DArray(&d_array[0],&channel_descriptor,volumeSize);
        CUDA_SYNC_CHECK();
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr((void *)h_array, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
        copyParams.dstArray = d_array[0];
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;

        cudaMemcpy3D(&copyParams);
        CUDA_SYNC_CHECK();
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = d_array[0];

        texDesc.addressMode[0]      = cudaAddressModeWrap;
        texDesc.addressMode[1]      = cudaAddressModeWrap;
        texDesc.filterMode          = cudaFilterModeLinear;
        texDesc.readMode            = cudaReadModeElementType;
        texDesc.normalizedCoords    = 1;
        texDesc.maxAnisotropy       = 1;
        texDesc.maxMipmapLevelClamp = 99;
        texDesc.minMipmapLevelClamp = 0;
        texDesc.mipmapFilterMode    = cudaFilterModePoint;
        texDesc.borderColor[0]      = 1.0f;
        texDesc.sRGB                = 0;
        cudaCreateTextureObject(&cudaTexture[0],&resDesc,&texDesc,nullptr);
        CUDA_SYNC_CHECK();
    }
    cudaSetDevice(0);
    free(h_array);
}

cudaTextureObject_t Volume::getTextureReference(const int iGPU) const{
    return cudaTexture[iGPU];
}

void Volume::getAabb(float results[6]){
    OptixAabb* aabb = reinterpret_cast<OptixAabb*>(results);

   float3 m_min = make_float3(position.x - worldSize.x/2.f, position.y - worldSize.y/2.f, position.z - worldSize.z/2.f);
   float3 m_max = make_float3(position.x + worldSize.y/2.f, position.y + worldSize.y/2.f, position.z + worldSize.z/2.f);

   *aabb = {
       m_min.x, m_min.y, m_min.z,
       m_max.x, m_max.y, m_max.z
   };
}

void Volume::getSbt(sbtData *sbt, const int iGPU){
//    sbt->id = id;
    sbt->volumeData.nbSlice = (int)slices.size();
    sbt->volumeData.tex = this->cudaTexture[iGPU];
    sbt->volumeData.size = worldSize;
    sbt->volumeData.center = position;
    sbt->volumeData.sizePixel = pixelSize;
    sbt->volumeData.isVisible = isVisible;

        //slices
    if( slices.size() != 0){
    	cudaSetDevice(iGPU);
    	sliceBuffer[iGPU].resize(slices.size() * sizeof(Slice));
    	sliceBuffer[iGPU].upload(slices.data(),slices.size());
    	sbt->volumeData.slices = (Slice*)sliceBuffer[iGPU].d_pointer();
    	cudaSetDevice(0);
    }
}
vec3f Volume::getWorldSize() const {return worldSize;}
vec3f Volume::getWorldCenter() const { return this->getPosition();}
vec3i Volume::getPixelSize() const {return pixelSize;}
cudaArray_t Volume::getTextureDevicePointer(const int iGPU) const { return d_array[iGPU];}




void Volume::resize(const vec3f &newDim){
    worldSize = newDim;
}


void Volume::addSlice(Slice slice){
    slices.push_back(slice);
}

void Volume::addSlice(vec3f p1, vec3f p2, vec3f p3){
    Slice s;
    s.p1 = p1;
    s.p2 = p2;
    s.p3 = p3;
    s.size = 0.5f;
    slices.push_back(s);
}

bool Volume::clearSlice(){
    slices.clear();
    return true;
}

void Volume::rmSlice(const int iSlice){
    slices.erase(slices.begin() + iSlice);
}
QJsonObject Volume::toJson(){
    QJsonObject json = AabbObject::toJson();

    json.insert("worldSize.X",worldSize.x);
    json.insert("worldSize.y",worldSize.y);
    json.insert("worldSize.z",worldSize.z);

    json.insert("pixelSizeX",pixelSize.x);
    json.insert("pixelSizeY",pixelSize.y);
    json.insert("pixelSizeZ",pixelSize.z);

    json.insert("type",type);
    return json;
}

