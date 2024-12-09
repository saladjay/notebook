#include <cuda_runtime.h>



__global__ void kernel() {
    // 使用外部的 __device__ 变量和函数

    printf("hello world\n");
}

int main() {
    // 启动内核
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}