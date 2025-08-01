#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Main program started.\n");

    void* ptr = malloc(100);  // 分配 100 字节
    if (ptr) {
        printf("Allocated 100 bytes at %p\n", ptr);
        free(ptr);  // 释放内存
        printf("Freed memory at %p\n", ptr);
    }

    printf("Main program finished.\n");
    return 0;
}