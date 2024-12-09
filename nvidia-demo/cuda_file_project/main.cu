// main.cu
#include <iostream>
#include <cuda_runtime.h>

// CUDA 内核函数
__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// 主机代码
void addWithCuda(int* c, const int* a, const int* b, size_t size) {
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // 分配 GPU 内存
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // 将输入数据从主机复制到 GPU
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // 将结果从 GPU 复制到主机
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    // 使用 CUDA 进行加法运算
    addWithCuda(c, a, b, arraySize);

    // 输出结果
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}