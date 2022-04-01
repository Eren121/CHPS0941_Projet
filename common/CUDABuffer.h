#ifndef CUDABUFFER_H
#define CUDABUFFER_H

#include "optix7.h"
#include <vector>
#include <assert.h>
#include <cuda.h>

 enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

/*! simple wrapper for creating, and managing a device-side CUDA
  buffer */
struct CUDABuffer {
inline CUdeviceptr d_pointer() const
{ return (CUdeviceptr)d_ptr; }

//! re-size buffer to given number of bytes
void resize(size_t size)
{
  if (d_ptr) free();
  alloc(size);
}

//! allocate to given number of bytes
void alloc(size_t size)
{
  assert(d_ptr == nullptr);
  this->sizeInBytes = size;
  CUDA_CHECK(cudaMalloc( (void**)&d_ptr, sizeInBytes));
}

//! free allocated memory
void free()
{
  CUDA_CHECK(cudaFree(d_ptr));
  d_ptr = nullptr;
  sizeInBytes = 0;
}

template<typename T>
void alloc_and_upload(const std::vector<T> &vt)
{
  alloc(vt.size()*sizeof(T));
  upload((const T*)vt.data(),vt.size());
}

template<typename T>
void upload(const T *t, size_t count)
{
  assert(d_ptr != nullptr);
  assert(sizeInBytes == count*sizeof(T));
  CUDA_CHECK(cudaMemcpyAsync(d_ptr, (void *)t,
                    count*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void upload(const T *t, size_t count, CUstream stream)
{
  assert(d_ptr != nullptr);
  assert(sizeInBytes == count*sizeof(T));
  CUDA_CHECK(cudaMemcpyAsync(d_ptr, (void *)t,
                    count*sizeof(T), cudaMemcpyHostToDevice, stream));
}
template<typename T>
void download(T *t, size_t count)
{
  assert(d_ptr != nullptr);
  assert(sizeInBytes == count*sizeof(T));
  CUDA_CHECK(cudaMemcpy((void *)t, d_ptr,
                    count*sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void download(T *t,size_t offset,size_t y_offset, size_t count, CUstream stream)
{
//  assert(d_ptr != nullptr);
//  assert(sizeInBytes == count*sizeof(T));
  CUDA_CHECK(cudaMemcpyAsync((void *)(t+offset), (void*)(&((uint32_t*)d_ptr)[y_offset]),
                    count*sizeof(T), cudaMemcpyDeviceToHost,stream));
}
size_t sizeInBytes { 0 };
void  *d_ptr { nullptr };
};


template <typename T>
void createOnDevice( const std::vector<T>& source, CUdeviceptr* destination )
{
    CUDA_CHECK(cudaMalloc( reinterpret_cast<void**>( destination ), source.size() * sizeof( T ) ) );
    copyToDevice( source, *destination );
}
template <typename T>
void copyToDevice( const T& source, CUdeviceptr destination )
{
    CUDA_CHECK(cudaMemcpy( reinterpret_cast<void*>( destination ), &source, sizeof( T ), cudaMemcpyHostToDevice ) );
}
#endif
