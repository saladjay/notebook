
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda_runtime_api.h>
// 定义函数指针类型
typedef cudaError_t (*malloc_func_t)(void **devPtr, size_t size);


// 获取原始函数
malloc_func_t original_malloc = NULL;

void load_func() __attribute__((constructor));
// void load_func() __attribute__((constructor))的含义是在执行main函数前，执行load_func这个函数，便于我们做一些准备工作。
// 显然这里的作用就是触发dlsym以实现查找第一个"cudaMalloc"函数符号的目的。
void load_func()
{
    original_malloc = (malloc_func_t)dlsym(RTLD_NEXT,"cudaMalloc");
    char *error_str;
    error_str = dlerror();
    if (error_str != NULL) {
        printf("%s\n", error_str);
    }
    printf("load func first f=%p\n",original_malloc);
 
}

static void initialize_original_functions() {
    if (!original_malloc) {
        original_malloc = (malloc_func_t)dlsym(RTLD_NEXT, "malloc");
    }
}

// 拦截 malloc
cudaError_t cudaMalloc(void **devPtr, size_t size) {

    printf("Intercepted malloc: allocating %zu bytes\n", size);
    cudaError_t error = original_malloc(devPtr, size);  // 调用原始的 malloc
    printf("end");
    return error;
}

