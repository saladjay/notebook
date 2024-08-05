#include <iostream>

#include "template.h"

template<int T>
split_impl<T>::split_impl(){
    if constexpr(T == 0){
        std::cout << "Splitting 0" << std::endl;
    }
    if constexpr(T == 1){
        std::cout << "Splitting 1" << std::endl;    
    }
}

template <int T>
split_impl<T>::~split_impl(){
    if constexpr(T == 0){
        std::cout << "Destructing 0" << std::endl;
    }
    if constexpr(T == 1){
        std::cout << "Destructing 1" << std::endl;
    }
}

template class split_impl<0>;
template class split_impl<1>;