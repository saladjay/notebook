/*
Copyright (C) 2019 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.

SPDX-License-Identifier: MIT
*/

#include <iostream>
#include <vector>
#include <tbb/tbb.h>
#include "ch01.h"

using ImagePtr = std::shared_ptr<ch01::Image>;

ImagePtr applyGamma(ImagePtr image_ptr, double gamma);
ImagePtr applyTint(ImagePtr image_ptr, const double *tints);
void writeImage(ImagePtr image_ptr);

class src_body{
    const int my_limit;
    int my_next_value;
public:
    src_body(int l) : my_limit(l), my_next_value(1){}
    int operator()(tbb::flow_control& fc){
        if(my_next_value <= my_limit){
            return my_next_value++;
        }else{
            fc.stop();
            return int();
        }
    }
};


void fig_1_10() {
  const double tint_array[] = {0.75, 0, 0};
  int sum = 0;
  tbb::flow::graph g;
  tbb::flow::broadcast_node<int> b(g);
  int i = 0;
  tbb::flow::function_node<int, int> squarer(g, tbb::flow::unlimited, [](const int& v){
    return v*v;
  });
  tbb::flow::function_node<int, int> cuber(g, tbb::flow::unlimited, [](const int& v){
    return v*v*v;
  });
  tbb::flow::function_node<int, int> summer(g, 1, [&](const int& v){
    return sum += v;
  });
//   make_edge(b, squarer);
//   make_edge(b, cuber);
  make_edge(squarer, summer);
  make_edge(cuber, summer);
  tbb::flow::input_node<int> src(g, src_body(10));
  make_edge(src, squarer);
  make_edge(src, cuber);

//   for ( int i = 1; i <= 10; ++i ) {
//     b.try_put(i);
//   }
  src.activate();
  g.wait_for_all();
  std::cout<< "Sum is "<<sum<<"\n";
} 

ImagePtr  applyGamma(ImagePtr image_ptr, double gamma) {
  auto output_image_ptr = 
    std::make_shared<ch01::Image>(image_ptr->name() + "_gamma", 
      ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  for ( int i = 0; i < height; ++i ) {
    for ( int j = 0; j < width; ++j ) {
      const ch01::Image::Pixel& p = in_rows[i][j]; 
      double v = 0.3*p.bgra[2] + 0.59*p.bgra[1] + 0.11*p.bgra[0];
      double res = pow(v, gamma);
      if(res > ch01::MAX_BGR_VALUE) res = ch01::MAX_BGR_VALUE;
      out_rows[i][j] = ch01::Image::Pixel(res, res, res);
    }
  }
  return output_image_ptr;
}

ImagePtr applyTint(ImagePtr image_ptr, const double *tints) {
  auto output_image_ptr = 
    std::make_shared<ch01::Image>(image_ptr->name() + "_tinted", 
      ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  for ( int i = 0; i < height; ++i ) {
    for ( int j = 0; j < width; ++j ) {
      const ch01::Image::Pixel& p = in_rows[i][j]; 
      std::uint8_t b = (double)p.bgra[0] + 
                       (ch01::MAX_BGR_VALUE-p.bgra[0])*tints[0];
      std::uint8_t g = (double)p.bgra[1] +  
                       (ch01::MAX_BGR_VALUE-p.bgra[1])*tints[1];
      std::uint8_t r = (double)p.bgra[2] + 
                       (ch01::MAX_BGR_VALUE-p.bgra[2])*tints[2];
      out_rows[i][j] = 
        ch01::Image::Pixel(
          (b > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : b,
          (g > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : g,
          (r > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : r
      );
    }
  }
  return output_image_ptr;
}

void writeImage(ImagePtr image_ptr) {
  image_ptr->write( (image_ptr->name() + ".bmp").c_str());
}

int main(int argc, char* argv[]) {
//   std::vector<ImagePtr> image_vector;

//   for ( int i = 2000; i < 20000000; i *= 10 ) 
//     image_vector.push_back(ch01::makeFractalImage(i));

//   // warmup the scheduler
//   tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
//     tbb::tick_count t0 = tbb::tick_count::now();
//     while ((tbb::tick_count::now() - t0).seconds() < 0.01);
//   });

//   tbb::tick_count t0 = tbb::tick_count::now();
//   fig_1_10(image_vector);
//   std::cout << "Time : " << (tbb::tick_count::now()-t0).seconds() 
//             << " seconds" << std::endl;
  fig_1_10();
  return 0;
}

