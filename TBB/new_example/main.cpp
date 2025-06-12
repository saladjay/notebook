#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <tbb/tbb.h>
int main() {
    tbb::parallel_invoke(
        []() { std::cout << "Hello " << std::this_thread::get_id() << std::endl; },
        []() { std::cout << "TBB! " << std::this_thread::get_id()<< std::endl; }
    );
    // return 0;

    std::vector<std::string> v = {"hello", "world", "oneapi", "tbb"};
    std::for_each(dpl::execution::par, v.begin(), v.end(), [](std::string &s){
        std::cout << s << std::this_thread::get_id() <<std::endl;
    });
    return 0;
}