#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

// 定义函数指针类型
typedef void* (*malloc_func_t)(size_t);
typedef void (*free_func_t)(void*);

// 获取原始函数
static malloc_func_t original_malloc = NULL;
static free_func_t original_free = NULL;

static void initialize_original_functions() {
    if (!original_malloc) {
        original_malloc = (malloc_func_t)dlsym(RTLD_NEXT, "malloc");
    }
    if (!original_free) {
        original_free = (free_func_t)dlsym(RTLD_NEXT, "free");
    }
}

// 拦截 malloc
void* malloc(size_t size) {
    initialize_original_functions();
    printf("Intercepted malloc: allocating %zu bytes\n", size);
    void* ptr = original_malloc(size);  // 调用原始的 malloc
    printf("Allocated at %p\n", ptr);
    return ptr;
}

// 拦截 free
void free(void* ptr) {
    initialize_original_functions();
    printf("Intercepted free: freeing memory at %p\n", ptr);
    original_free(ptr);  // 调用原始的 free
}