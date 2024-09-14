#include <cstdio>

// #include <cstdio>
 
int a() { return std::puts("a"); }
int b() { return std::puts("b"); }
int c() { return std::puts("c"); }
 
void z(int, int, int) {}
 
int main()
{
    z(a(), b(), c());       // 允许全部 6 种输出排列
    return a() + b() + c(); // 允许全部 6 种输出排列
}