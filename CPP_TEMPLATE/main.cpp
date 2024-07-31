#include <iostream>
#include "template.h"
int main() {    
    auto a = split_impl();
    auto b = split_impl<1>();
}