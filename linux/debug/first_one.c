/*
文件名:first_one.c
编译成动态库:gcc -fpic --shared first_one.c -o libfirst_one.so
*/
 
#include <stdio.h>
void print_message()
{
    printf("the first lib~~\n");
}
void first()
{
    printf("init first\n");
}