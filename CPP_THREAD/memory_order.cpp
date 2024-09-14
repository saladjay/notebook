#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <cassert>
 
std::atomic<int> cnt = {0};
 
void f()
{
    for (int n = 0; n < 1000; ++n)
        cnt.fetch_add(1, std::memory_order_relaxed);
}
 
int memory_order_relaxed()
{
    std::vector<std::thread> v;
    for (int n = 0; n < 10; ++n)
        v.emplace_back(f);
    for (auto& t : v)
        t.join();
    std::cout << "the final value of cnt is " << cnt << '\n';
}

std::atomic<std::string*> ptr;
int data;
 
void producer()
{
    std::string* p = new std::string("Hello");
    data = 42;
    ptr.store(p, std::memory_order_release);
}
 
void consumer()
{
    std::string* p2;
    while (!(p2 = ptr.load(std::memory_order_acquire)));
    assert(*p2 == "Hello"); // 绝无问题
    assert(data == 42); // 绝无问题
}
 
int Release_Acquire_ordering()
{
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join(); t2.join();
}

std::vector<int> data2;
std::atomic<int> flag2 = {0};
 
void thread_1()
{
    data2.push_back(42);
    flag2.store(1, std::memory_order_release);
}
 
void thread_2()
{
    int expected=1;
    // memory_order_relaxed 是可以的，因为这是一个 RMW 操作
    // 而 RMW（以任意定序）跟在释放之后将组成释放序列
    while (!flag2.compare_exchange_strong(expected, 2, std::memory_order_relaxed))
    {
        expected = 1;
    }
}
 
void thread_3()
{
    while (flag2.load(std::memory_order_acquire) < 2);
    // 如果我们从 atomic flag 中读到 2，将看到 vector 中储存 42
    assert(data2.at(0) == 42); //决不出错
}
 
int main()
{
    std::thread a(thread_1);
    std::thread b(thread_2);
    std::thread c(thread_3);
    a.join(); b.join(); c.join();
}