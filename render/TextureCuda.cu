#include "TextureCuda.h"
#include <climits>
#include <utility>

__global__ void k_testLut(cudaTextureObject_t lut) {

    printf("lut (device):\n");
    const int max = 1000;
    for(int i = 0; i < max; i++) {
        const float f = static_cast<float>(i) / max;
        
        //const float4 c = tex2D<float4>(optixLaunchParams.renderingTypeOptions.lut, f, 0.0f);
        //printf("lut(x=%f) = (%f, %f, %f, %f)\n", f, c.x, c.y, c.z, c.w);
        
        const float4 c = tex2D<float4>(lut, f, 0.0f);
        printf("lut(x=%f) = (%f, %f, %f, %f)\n", f * 255, c.x * 255, c.y * 255, c.z * 255, c.w * 255);
    }
}

void testLut(cudaTextureObject_t lut) {
    k_testLut<<<1, 1>>>(lut);
}

TextureCuda::TextureCuda(const uchar4* data, size_t size)
{
    const size_t width = size;
    const size_t height = 1;
    const size_t nBitsPerTexelPerChannel = sizeof(unsigned char) * CHAR_BIT; // Pour expliciter, = 8 ici
    const cudaChannelFormatKind format = cudaChannelFormatKindUnsigned;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc(nBitsPerTexelPerChannel, nBitsPerTexelPerChannel, nBitsPerTexelPerChannel, nBitsPerTexelPerChannel, format);

    CUDA_CHECK(cudaMallocArray(&m_d_array, &desc, width, height));

    const size_t pitch = sizeof(uchar4) * width;
    const size_t nBytesWidth = pitch;
    CUDA_CHECK(cudaMemcpy2DToArray(m_d_array, 0, 0, data, pitch, nBytesWidth, height, cudaMemcpyHostToDevice));
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_d_array;

    const cudaTextureAddressMode mode = cudaAddressModeClamp; // Limiter aux bords du gradient
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = mode;
    texDesc.addressMode[1] = mode;
    texDesc.filterMode = cudaFilterModeLinear; // Interpolation linéaire

    // On envoit des uchar lors de l'initialisation mais on souhaite lire des floats dans [0;1]
    // lors du texture fetch
    texDesc.readMode = cudaReadModeNormalizedFloat;

    // Normaliser les coordonnées (comme UV en OpenGL)
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&m_textureObj, &resDesc, &texDesc, nullptr));
}

TextureCuda::TextureCuda(TextureCuda&& right)
{
    swap(*this, right);
}

TextureCuda& TextureCuda::operator=(TextureCuda&& right)
{
    swap(*this, right);
    return *this;
}

TextureCuda::~TextureCuda()
{
    if(m_textureObj) {
        CUDA_CHECK(cudaDestroyTextureObject(m_textureObj));
    }

    if(m_d_array) {
        CUDA_CHECK(cudaFreeArray(m_d_array));
    }
}

void swap(TextureCuda& lhs, TextureCuda& rhs)
{
    using std::swap;
    swap(lhs.m_textureObj, rhs.m_textureObj);
    swap(lhs.m_d_array, rhs.m_d_array);
}