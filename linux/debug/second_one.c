/*
文件名：second_one.c
编译动态库: gcc -fpic --shared second_one.c -o libsecond_one.so
*/
 
#include <stdio.h>
void print_message()
{
    printf("the second lib~~\n");
}
 
void second()
{
    printf("init second \n");
}