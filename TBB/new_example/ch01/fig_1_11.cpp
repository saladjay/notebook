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
void writeImage(ImagePtr image_ptr);

class src_body{
    const std::vector<ImagePtr> my_limit;
    int my_next_value;
public:
    src_body(std::vector<ImagePtr> l) : my_limit(l), my_next_value(0){}
    ImagePtr operator()(tbb::flow_control& fc){
        if(my_next_value < my_limit.size()){
          // std::cout<<"new image start on "<<std::this_thread::get_id()<<std::endl;
            return my_limit[my_next_value++];
        }else{
            fc.stop();
            return nullptr;
        }
    }
};

ImagePtr applyGamma(ImagePtr image_ptr, double gamma) {
  auto output_image_ptr = 
    std::make_shared<ch01::Image>(image_ptr->name() + "_gamma", 
      ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  tbb::parallel_for( 0, height, 
    [&in_rows, &out_rows, width, gamma](int i) {
      for ( int j = 0; j < width; ++j ) {
        const ch01::Image::Pixel& p = in_rows[i][j]; 
        double v = 0.3*p.bgra[2] + 0.59*p.bgra[1] + 0.11*p.bgra[0];
        double res = pow(v, gamma);
        if(res > ch01::MAX_BGR_VALUE) res = ch01::MAX_BGR_VALUE;
        out_rows[i][j] = ch01::Image::Pixel(res, res, res);
      }
    }
  );
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

  tbb::parallel_for( 0, height, 
    [&in_rows, &out_rows, width, tints](int i) {
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
  );
  return output_image_ptr;
}

void fig_1_11(std::vector<ImagePtr>& image_vector) {
  const double tint_array[] = {0.75, 0, 0};

  tbb::flow::graph g;

  int i = 0;
  tbb::flow::input_node<ImagePtr> src(g, src_body(image_vector));

  tbb::flow::function_node<ImagePtr, ImagePtr> gamma(g, 
    tbb::flow::unlimited,
    [] (ImagePtr img) -> ImagePtr {
      return applyGamma(img, 1.4);  
    }
  );

  tbb::flow::function_node<ImagePtr, ImagePtr> tint(g, 
    tbb::flow::unlimited,
    [tint_array] (ImagePtr img) -> ImagePtr {
      return applyTint(img, tint_array);
    }
  );

  tbb::flow::function_node<ImagePtr> write(g, 
    tbb::flow::unlimited,
    [] (ImagePtr img) {
      writeImage(img);
    }
  );

  tbb::flow::make_edge(src, gamma);
  tbb::flow::make_edge(gamma, tint);
  tbb::flow::make_edge(tint, write);
  src.activate();
  g.wait_for_all();
} 

void writeImage(ImagePtr image_ptr) {
  image_ptr->write( (image_ptr->name() + ".bmp").c_str());
}

int main(int argc, char* argv[]) {
  std::vector<ImagePtr> image_vector;

  for ( int i = 2000; i < 20000000; i *= 10 ) 
    image_vector.push_back(ch01::makeFractalImage(i));

  // warmup the scheduler
  tbb::parallel_for(0, tbb::info::default_concurrency(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.01);
  });

  tbb::tick_count t0 = tbb::tick_count::now();
  fig_1_11(image_vector);
  std::cout << "Time : " << (tbb::tick_count::now()-t0).seconds() 
            << " seconds" << std::endl;
  return 0;
}

