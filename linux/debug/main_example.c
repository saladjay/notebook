/*
文件名：main.c
编译动态库: gcc -c main.c
*/
#include <stdio.h>
void print_message();
void first();
void second();
int main()
{
    first();
    second();
    print_message();
    return 0;
}