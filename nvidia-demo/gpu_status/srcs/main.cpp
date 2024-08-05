#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

int main(){
    size_t free, total;
    cudaError_t cudaStatus;

    // 初始化CUDA
    cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaFree failed!\\n");
        return 1;
    }

    // 获取显存使用情况
    cudaStatus = cudaMemGetInfo(&free, &total);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed!\\n");
        return 1;
    }

    printf("Free memory: %zu bytes\n", free);
    printf("Total memory: %zu bytes\n", total);

    size_t used = total - free;
    printf("Used memory: %zu bytes\n", used);

    void* ptr{nullptr};
    cudaMalloc((void**)&ptr,1024*1024*1024*sizeof(char));

        // 获取显存使用情况
    cudaStatus = cudaMemGetInfo(&free, &total);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed!\\n");
        return 1;
    }

    printf("Free memory: %zu bytes\n", free);
    printf("Total memory: %zu bytes\n", total);

    used = total - free;
    printf("Used memory: %zu bytes\n", used);
    return 0;
}