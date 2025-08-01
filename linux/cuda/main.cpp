#include <iostream>
#include <cuda_runtime_api.h>
cudaError_t cudaMalloc(void **devPtr, size_t size);
int main(){
    cudaSetDevice(0);
    std::cout<<__LINE__<<std::endl;
    float *ptr{nullptr};
    std::cout<<__LINE__<<std::endl;
    auto error = cudaMalloc((void**)&ptr, sizeof(float) * 10);
    std::cout<<cudaGetErrorString(error)<<std::endl;
    std::cout<<__LINE__<<std::endl;
}