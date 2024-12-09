#include <iostream>
#include <thread>
#include <atomic>
#include <assert.h>
#include <chrono>
#include <vector>

/**
 * 编译器优化而产生的指令乱序，cpu指令流水线也会产生指令乱序，总体来讲，编译器优化层面会产生的指令乱序，
 * cpu层面也会的指令流水线优化产生乱序指令。当然这些乱序指令都是为了同一个目的，优化执行效率.
 * happens-before:按照程序的代码序执行
 */
// 6种memory_order 主要分成3类，relaxed(松弛的内存序)，sequential_consistency（内存一致序）,acquire-release(获取-释放一致性)

// relaxed memory model
// 没有顺序一致性的要求，也就是说同一个线程的原子操作还是按照happens-before关系，但不同线程间的执行关系是任意。

std::atomic<bool> x, y;
std::atomic<int> z;

void write_x_then_y()
{
    x.store(true, std::memory_order_relaxed); // 1
    y.store(true, std::memory_order_relaxed); // 2
}

void read_y_then_x()
{
    while (!y.load(std::memory_order_relaxed))
        ; // 3
    if (x.load(std::memory_order_relaxed))
    { // 4
        ++z;
    }
}

int experment_relaxed()
{
    x = false;
    y = false;
    z = 0;
    std::thread t1(write_x_then_y);
    std::thread t2(read_y_then_x);
    t1.join();
    t2.join();
    if (z.load() != 0)
    {
        std::cout << "relaxed memory model: success" << std::endl;
    }
    else
    {
        std::cout << "relaxed memory model: fail" << std::endl;
    }
    return 0;
}
// sequential consistency(内存一致性)
// 同一个线程的原子操作按照happens-before关系执行，不同线程间的执行关系是顺序一致的。
/** 
 * 这个是以牺牲优化效率，来保证指令的顺序一致执行，相当于不打开编译器优化指令，按照正常的指令序执行(happens-before)，
 * 多线程各原子操作也会Synchronized-with，（譬如atomic::load()需要等待atomic::store()写下元素才能读取，同步过程），
 * 当然这里还必须得保证一致性，读操作需要在“一个写操作对所有处理器可见”的时候才能读，适用于基于缓存的体系结构。
 * */
std::vector<int> data;
std::atomic_bool data_ready(false);

// 线程1
void writer_thread()
{
    data.push_back(10); // #1：对data的写操作
    data_ready = true;  // #2：对data_ready的写操作
}

// 线程2
void reader_thread()
{
    while (!data_ready.load()) // #3：对data_ready的读操作
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "data is " << data[0] << "\n"; // #4：对data的读操作
}

int experment_sequential_consistency()
{
    data.clear();
    data_ready = false;
    std::thread t1(writer_thread);
    std::thread t2(reader_thread);
    t1.join();
    t2.join();

    return 0;
}

int main()
{
    experment_relaxed();

    experment_sequential_consistency();
    return 0;
}