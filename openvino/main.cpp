#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <regex>

// clang-format off
#include <openvino/openvino.hpp>
#include "openvino/openvino.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/core/type/element_type.hpp"
// E:\3rd_party\openvino_compile\OpenVINO\runtime\include\openvino\core\type\element_type.hpp

#include "Block.h"  
#define slog std
#define info cout

void printInputAndOutputsInfo(const ov::Model& network) {
    std::cout << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

// using regex to extract shape from string
std::vector<size_t> get_shape(std::string shape){
    std::vector<size_t> res;
    std::regex pattern("\\d+");
    std::smatch result;
    while(std::regex_search(shape, result, pattern)){
        res.push_back(std::stoi(result[0]));
        shape = result.suffix();
    }
    return res;
}

template<typename T, typename U>
std::vector<T> NCHWtoNHWC(const std::vector<T>& input, U N, U C, U H, U W) {
    std::vector<T> output(N * H * W * C);
    for (U n = 0; n < N; ++n) {
        for (U h = 0; h < H; ++h) {
            for (U w = 0; w < W; ++w) {
                for (U c = 0; c < C; ++c) {
                    output[n * H * W * C + h * W * C + w * C + c] = input[n * C * H * W + c * H * W + h * W + w];
                }
            }
        }
    }
    return output;
}

template<typename T>
std::vector<T> NHWCtoNCHW(const std::vector<T>& input, int N, int C, int H, int W) {
    std::vector<T> output(N * C * H * W);
    const o_1st_size = C * H * W;
    const o_2nd_size = H * W;
    const o_3rd_size = W;

    const i_1st_size = H * W * C;
    const i_2nd_size = W * C;
    const i_3rd_size = C;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    output[n * o_1st_size + c * o_2nd_size + h * o_3rd_size + w] = 
                     input[n * i_1st_size + h * i_2nd_size + w * i_3rd_size + c];
                }
            }
        }
    }
    return output;
}

std::tuple<std::vector<size_t>, std::vector<float>> read_tensor(std::string path){
    // Read tensor data
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
        return {{},{}};
    }
    int32_t shape_size_len;
    file.read(reinterpret_cast<char*>(&shape_size_len), sizeof(shape_size_len));
    
    char* shape_size = new char[shape_size_len];
    file.read(shape_size, shape_size_len);
    std::string shape_str(shape_size, shape_size_len);
    std::vector<size_t> shape = get_shape(shape_str);
    size_t volume = std::accumulate(shape.begin(), shape.end(), 1ull, std::multiplies<size_t>());

    std::vector<float> data(volume);
    file.read(reinterpret_cast<char*>(data.data()), volume * sizeof(float));

    delete[] shape_size;
    return {shape, data};
}

std::map<std::string, Weights> read_weights(std::string path){
    using python_int = int32_t;
	using python_float = double;
	using pytorch_default_dtype = float;
    // Read weights data
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::map<std::string, Weights> weights_map;
    python_int nb_layer;
	file.read(reinterpret_cast<char *>(&nb_layer), sizeof(nb_layer));
    for (python_int i = 0; i < nb_layer; ++i)
	{
        Weights weights;
		python_int nb_name_length;
		file.read(reinterpret_cast<char *>(&nb_name_length), sizeof(nb_name_length));

        char *weight_name = new char[nb_name_length];
		file.read(weight_name, nb_name_length * sizeof(char));
		std::stringstream weight_name_stream;
		weight_name_stream.write(weight_name, nb_name_length);

        python_int nb_weight_length;
		file.read(reinterpret_cast<char *>(&nb_weight_length), sizeof(nb_weight_length));

        pytorch_default_dtype *val = new pytorch_default_dtype[nb_weight_length];
		file.read(reinterpret_cast<char *>(val), nb_weight_length * sizeof(pytorch_default_dtype));

        weights.values = val;
        weights.count = nb_weight_length;
        weights_map[weight_name_stream.str()] = weights;
        std::cout<<"read weights: "<< weight_name_stream.str()<<std::endl;
        std::cout<<"len: "<<nb_weight_length<<std::endl;
        delete[] weight_name;
    }
    return weights_map;
}

void infer(std::shared_ptr<ov::Model> &model, std::vector<uint64_t> &pytorch_input_shape, std::vector<float> &input_data,
            std::vector<uint64_t> &pytorch_output_shape, std::vector<float> &output_data){
    // -------- Step 1. Read input data --------
    using namespace ov;
    ov::Shape input_shape = model->input().get_shape();
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    const ov::Shape output_shape = model->output().get_shape();
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;

    // -------- Step 3. Apply preprocessing --------
    const Layout tensor_layout{"NHWC"};

    // apply preprocessing
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    // 1) InputInfo() with no args assumes a model has a single input
    ov::preprocess::InputInfo& input_info = ppp.input();
    // 2) Set input tensor information:
    // - layout of data is 'NHWC'
    // - precision of tensor is supposed to be 'u8'
    input_info.tensor().set_layout(tensor_layout).set_element_type(element::f32);
    // 3) Here we suppose model has 'NCHW' layout for input
    input_info.model().set_layout("NCHW");

    // 4) Once the build() method is called, the preprocessing steps
    // for layout and precision conversions are inserted automatically
    model = ppp.build();

    // Set batch size using images count
    const size_t batch_size = pytorch_input_shape[0];

    // -------- Step 4. Reshape a model to new batch size --------
    // Setting batch size using image count
    ov::set_batch(model, batch_size);
    std::cout << "Batch size is " << std::to_string(batch_size) << std::endl;
    printInputAndOutputsInfo(*model);

    // -------- Step 5. Compiling model for the device --------
    const std::string device_name = "GPU";
    std::cout << "Compiling a model for the " << device_name << " device" << std::endl;
    ov::Core core;


    ov::CompiledModel compiled_model;
    try
    {
        compiled_model = core.compile_model(model, device_name);
    }
    catch(const std::exception& e)
    {        
        std::cout<<__FILE__<<":"<<__LINE__<<std::endl;std::cerr << e.what() << '\n';
    }
        
    // -------- Step 6. Create infer request --------
    std::cout << "Create infer request" << std::endl;
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    // -------- Step 7. Combine multiple input images as batch --------
    std::cout << "Combine images in batch and set to input tensor" << std::endl;
    ov::Tensor input_tensor = infer_request.get_input_tensor();

    // Iterate over all input images and copy data to input tensor
    // for (size_t image_id = 0; image_id < digits.size(); ++image_id) {
    const size_t image_size = shape_size(model->input().get_shape()) / batch_size;
    std::cout<<"image_size: "<<image_size<<std::endl;

    std::vector<float> transpose_input_data = NCHWtoNHWC(input_data, pytorch_input_shape[0], pytorch_input_shape[1],
                                pytorch_input_shape[2], pytorch_input_shape[3]);
    std::memcpy(input_tensor.data<float>(), transpose_input_data.data(), image_size * batch_size * sizeof(float));
            
    std::cout<<"input tensor data type: "<<input_tensor.get_element_type()<<std::endl;
    std::cout<<"input tensor shape: "<<input_tensor.get_shape().to_string()<<std::endl;

    // -------- Step 8. Do sync inference --------
    slog::info << "Start sync inference" << slog::endl;
    infer_request.infer();

    // -------- Step 9. Process output --------
    slog::info << "Processing output tensor" << slog::endl;
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    std::cout<<"output_tensor: "<<output_tensor.get_element_type()<<std::endl;

    using tensor_type = ov::fundamental_type_for<ov::element::Type_t::f32>;
    auto batchData = output_tensor.data<float>();
    auto openvino_output_shape = output_tensor.get_shape();
    for(size_t i{0};i<4;++i){
        if(openvino_output_shape[i] != pytorch_output_shape[i]){
            std::cout<<"output size is not equal at dim "<< i <<"\n";
            std::cout<<"pytorch shape:"<<Shape(pytorch_output_shape).to_string()<<"\n";
            std::cout<<"openvino shape:"<<openvino_output_shape.to_string()<<std::endl;
        }
    }
    auto output_len = std::accumulate(openvino_output_shape.begin(), openvino_output_shape.end(), 1ull, std::multiplies<size_t>());
    float biggest_diff{0};
    size_t diff_count{0};
    for(size_t index{0};index<output_len;index++){
        auto diff_offset = std::abs(batchData[index]-output_data[index]);
        biggest_diff = std::max(diff_offset, biggest_diff);

        if(diff_offset > 0.002){
            diff_count++;
            // std::cout<<"output not equal at "<<index<<std::endl;
            // std::cout<<"openvino: "<<batchData[index]<<" pytorch: "<<output_data[index]<<std::endl;
            // break;
        }
    }
    std::cout<<"biggest difference is "<<biggest_diff<<std::endl;
    std::cout<<"diff count is "<<diff_count<<std::endl;
}

void testConv(){
    auto [input_shape, input_data] = read_tensor("D:\\github\\ultralytics\\output\\input.bin");
    auto [output_shape, output_data] = read_tensor("D:\\github\\ultralytics\\output\\conv.bin");
    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape(input_shape));
    auto weights_map = read_weights("D:\\github\\ultralytics\\output\\conv.weights");
    auto layer = convBnSiLu(paramNode->output(0), weights_map, 8, 3, 1, 1, "model");
    layer->get_output_tensor(0).set_names({"output_tensor"});
    auto result = std::make_shared<ov::opset15::Result>(layer->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result, ov::ParameterVector{paramNode}, "model");
    infer(model, input_shape, input_data, output_shape, output_data);
}

void testBottleneck(){
    auto [input_shape, input_data] = read_tensor("D:\\github\\ultralytics\\output\\input.bin");
    auto [output_shape, output_data] = read_tensor("D:\\github\\ultralytics\\output\\bottleneck.bin");
    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape(input_shape));
    auto weights_map = read_weights("D:\\github\\ultralytics\\output\\bottleneck.weights");
    auto layer = bottleneck(paramNode->output(0), weights_map, 4, 4, true, 1.0, "model");
    layer->get_output_tensor(0).set_names({"output_tensor"});
    auto result = std::make_shared<ov::opset15::Result>(layer->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result, ov::ParameterVector{paramNode}, "model");
    infer(model, input_shape, input_data, output_shape, output_data);
}

void testC2f(){
    auto [input_shape, input_data] = read_tensor("D:\\github\\ultralytics\\output\\input.bin");
    auto [output_shape, output_data] = read_tensor("D:\\github\\ultralytics\\output\\c2f.bin");
    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape(input_shape));
    auto weights_map = read_weights("D:\\github\\ultralytics\\output\\c2f.weights");
    auto layer = C2F(paramNode->output(0), weights_map, 4, 8, 3, true, 0.5, "model");
    layer->get_output_tensor(0).set_names({"output_tensor"});
    auto result = std::make_shared<ov::opset15::Result>(layer->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result, ov::ParameterVector{paramNode}, "model");
    infer(model, input_shape, input_data, output_shape, output_data);
}


void testSPPF(){
    auto [input_shape, input_data] = read_tensor("D:\\github\\ultralytics\\output\\input.bin");
    auto [output_shape, output_data] = read_tensor("D:\\github\\ultralytics\\output\\sppf.bin");
    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape(input_shape));
    auto weights_map = read_weights("D:\\github\\ultralytics\\output\\sppf.weights");
    auto layer = SPPF(paramNode->output(0), weights_map, 4, 4, 5, "model");
    layer->get_output_tensor(0).set_names({"output_tensor"});
    auto result = std::make_shared<ov::opset15::Result>(layer->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result, ov::ParameterVector{paramNode}, "model");
    infer(model, input_shape, input_data, output_shape, output_data);
}

void testUpsample(){
    auto [input_shape, input_data] = read_tensor("D:\\github\\ultralytics\\output\\input.bin");
    auto [output_shape, output_data] = read_tensor("D:\\github\\ultralytics\\output\\upsample.bin");
    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape(input_shape));
    auto weights_map = read_weights("D:\\github\\ultralytics\\output\\upsample.weights");
    auto layer = upsample(paramNode->output(0), std::vector<float>{1.0, 1.0, 2.0, 2.0});
    layer->get_output_tensor(0).set_names({"output_tensor"});
    auto result = std::make_shared<ov::opset15::Result>(layer->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result, ov::ParameterVector{paramNode}, "model");
    infer(model, input_shape, input_data, output_shape, output_data);
}

void testDFL(){
    auto [input_shape, input_data] = read_tensor("D:\\github\\ultralytics\\output\\dfl.bin");
    auto [output_shape, output_data] = read_tensor("D:\\github\\ultralytics\\output\\dfl.bin");
    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape(input_shape));
    auto weights_map = read_weights("D:\\github\\ultralytics\\output\\dfl.weights");
    auto layer = DFL(paramNode->output(0), weights_map, 16, "model");
    layer->get_output_tensor(0).set_names({"output_tensor"});
    auto result = std::make_shared<ov::opset15::Result>(layer->output(0));
    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result, ov::ParameterVector{paramNode}, "model");
    infer(model, input_shape, input_data, output_shape, output_data);
}

void testDetect(){
    
}

int main(){
    try
    {
        testDFL();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
    return 0;
}

int main1() {
    using namespace ov;
    try{
    // Create a vector of input shapes
    std::vector<ptrdiff_t> padBegin{1, 1}; // top, left
    std::vector<ptrdiff_t> padEnd{1, 1}; // bottom, right

    auto paramNode = std::make_shared<ov::opset15::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 4, 4, 4}));
    
    // std::cout<<paramNode->get_output_partial_shape(0).to_string()<<std::endl;
    // std::cout<<paramNode->output(0).get_partial_shape()[1].get_length()<<std::endl;
    // Create a convolution node with padding
    auto convFirstShape = ov::Shape{3, 4, 3, 3}; // output_c, input_c, kernel_h, kernel_w
    // std::vector<float> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    float data[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,};
    // float data[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    //                         2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    //                         3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,};

    auto convolutionFirstConstantNode = std::make_shared<opset15::Constant>(element::Type_t::f32, convFirstShape, data);
    std::cout<<"values:"<<std::endl;
    for(int i{0};i<27;i++){
        std::cout<< convolutionFirstConstantNode->convert_value_to_string(i)<<std::endl;
    }

    auto convolutionNodeFirst = std::make_shared<opset15::Convolution>(paramNode->output(0),
                                                                       convolutionFirstConstantNode->output(0),
                                                                       Strides({1, 1}),
                                                                       CoordinateDiff(padBegin),
                                                                       CoordinateDiff(padEnd),
                                                                       Strides({1, 1}));
    convolutionNodeFirst->get_input_tensor(0).set_names({"input"});
    std::cout<<"convolutionNodeFirst: "<<convolutionNodeFirst->get_output_partial_shape(0).to_string()<<std::endl;
                                 
    // --------------Add--------------
    
        auto biasShape = ov::Shape{1, 3, 1, 1};
        std::vector<float> biasData{0.0, 0.0, 0.0};
        // for(int i = 0; i < 3; i++){
        //     for(int j = 0; j < 8*8; j++)
        //     {
        //         biasData[i*8*8+j] = i;
        //     }
        // }
        auto biasConstantNode = std::make_shared<opset15::Constant>(element::Type_t::f32, biasShape, biasData);

        auto addNode = std::make_shared<opset15::Add>(convolutionNodeFirst->output(0), biasConstantNode->output(0));
        std::cout<<"addNode: "<<addNode->get_output_partial_shape(0).to_string()<<std::endl;


    // 
    // --------------BatchNormalization--------------
    auto bnShape = ov::Shape{3};
    float scale_values[3] = {1, 1 ,1};
    auto bnScale = std::make_shared<opset15::Constant>(element::Type_t::f32, bnShape, scale_values);
    float bias_values[3] = {0.0 ,0.0 ,0.0};
    auto bnB = std::make_shared<opset15::Constant>(element::Type_t::f32, bnShape, bias_values);
    float mean_values[3] = {10, 10, 10};
    auto bnMean = std::make_shared<opset15::Constant>(element::Type_t::f32, bnShape, mean_values);
    float var_values[3] = {4, 4, 4};
    auto bnVar = std::make_shared<opset15::Constant>(element::Type_t::f32, bnShape, var_values);
    auto bn = std::make_shared<opset15::BatchNormInference>(addNode->output(0), bnScale->output(0), bnB->output(0), bnMean->output(0), bnVar->output(0), 1e-5);

    
    auto sigmoid = std::make_shared<opset15::Sigmoid>(bn->output(0));

    // Create parameters for Slice operation
    auto shape = addNode->get_output_shape(0);
    std::vector<uint64_t> start_indices{0, 0, 0, 0};                        // Start from [0,1,0,0]
    std::vector<uint64_t> end_indices{1, 2, shape[2], shape[3]}; // End at [1,4,8,1]
    std::vector<uint64_t> strides{1, 1, 1, 1};                              // Use stride 2 along axis 2
    auto split1 = std::make_shared<opset15::Slice>(addNode->output(0),
                                          opset15::Constant::create(ov::element::u64, {4}, start_indices), 
                                          opset15::Constant::create(ov::element::u64, {4}, end_indices), 
                                          opset15::Constant::create(ov::element::u64, {4}, strides));


    std::shared_ptr<ov::op::Op> aa = split1;
    aa->get_output_tensor(0).set_names({"output_tensor"});
    auto result_full = std::make_shared<opset15::Result>(aa->output(0));

    // addNode->get_output_tensor(0).set_names({"output_tensor"});
    // auto result_full = std::make_shared<opset15::Result>(addNode->output(0));

    std::shared_ptr<ov::Model> model =
        std::make_shared<ov::Model>(result_full, ov::ParameterVector{paramNode}, "lenet");
    

    ov::Shape input_shape = model->input().get_shape();
    std::cout << "Input shape: " << input_shape.to_string() << std::endl;
    const ov::Shape output_shape = model->output().get_shape();
    std::cout << "Output shape: " << output_shape.to_string() << std::endl;

     // -------- Step 3. Apply preprocessing --------
        const Layout tensor_layout{"NHWC"};

        // apply preprocessing
        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

        // 1) InputInfo() with no args assumes a model has a single input
        ov::preprocess::InputInfo& input_info = ppp.input();
        // 2) Set input tensor information:
        // - layout of data is 'NHWC'
        // - precision of tensor is supposed to be 'u8'
        input_info.tensor().set_layout(tensor_layout).set_element_type(element::f32);
        // 3) Here we suppose model has 'NCHW' layout for input
        input_info.model().set_layout("NCHW");

        // 4) Once the build() method is called, the preprocessing steps
        // for layout and precision conversions are inserted automatically
        model = ppp.build();

        // Set batch size using images count
        const size_t batch_size = 1;

        // -------- Step 4. Reshape a model to new batch size --------
        // Setting batch size using image count
        ov::set_batch(model, batch_size);
        std::cout << "Batch size is " << std::to_string(batch_size) << std::endl;
        printInputAndOutputsInfo(*model);

        // -------- Step 5. Compiling model for the device --------
        const std::string device_name = "GPU";
        std::cout << "Compiling a model for the " << device_name << " device" << std::endl;
        ov::Core core;
        // std::cout << "Device info: " << std::endl;
        // auto versions = core.get_versions(device_name);
        // for(auto&& version : versions){
        //     std::cout << version.first << " : " << version.second.buildNumber << " : " << version.second.description << std::endl;
        // }
        // std::cout << core.get_versions(device_name) << std::endl;

        ov::CompiledModel compiled_model;
        try
        {
            compiled_model = core.compile_model(model, device_name);
        }
        catch(const std::exception& e)
        {
            
            std::cout<<__FILE__<<":"<<__LINE__<<std::endl;std::cerr << e.what() << '\n';
        }
        
        

        // -------- Step 6. Create infer request --------
        std::cout << "Create infer request" << std::endl;
        ov::InferRequest infer_request = compiled_model.create_infer_request();
        // -------- Step 7. Combine multiple input images as batch --------
        std::cout << "Combine images in batch and set to input tensor" << std::endl;
        ov::Tensor input_tensor = infer_request.get_input_tensor();

        // Iterate over all input images and copy data to input tensor
        // for (size_t image_id = 0; image_id < digits.size(); ++image_id) {
            const size_t image_size = shape_size(model->input().get_shape()) / batch_size;
            std::cout<<"image_size: "<<image_size<<std::endl;
        float data_batch[64];
        // for(size_t i{0};i<64;i++){
        //     if (i<16)
        //         data_batch[i] = 1;
        //     else if (i<32)
        //         data_batch[i] = 2;
        //     else if (i<48)
        //         data_batch[i] = 3;
        //     else
        //         data_batch[i] = 4;
        // }
        for(size_t i{0};i<64;i++){
            data_batch[i] = float(i % 4);
        }
        for (size_t  i = 0; i < 64; i++)
        {

            
            std::cout<<float(data_batch[i])<<",";
        }
        std::cout<<std::endl;
        
            std::memcpy(input_tensor.data<float>(), data_batch, image_size * sizeof(float));
            std::cout<<"input tensor data type: "<<input_tensor.get_element_type()<<std::endl;
            std::cout<<"input tensor shape: "<<input_tensor.get_shape().to_string()<<std::endl;
            // std::memcpy(input_tensor.data<std::uint8_t>() + image_id * image_size, digits[image_id], image_size);
        // }

        // -------- Step 8. Do sync inference --------
        slog::info << "Start sync inference" << slog::endl;
        int count_infer = 1000;
        while(count_infer > 0){
            infer_request.infer();
            count_infer--;
        }

                // -------- Step 9. Process output --------
        slog::info << "Processing output tensor" << slog::endl;
        ov::Tensor output_tensor = infer_request.get_output_tensor();
        std::cout<<"output_tensor: "<<output_tensor.get_element_type()<<std::endl;

    try
    {
        using tensor_type = ov::fundamental_type_for<ov::element::Type_t::f32>;
        auto batchData = output_tensor.data<float>();
        auto shape = output_tensor.get_shape();
        std::cout<<"output_tensor shape:"<<shape.to_string()<<std::endl;
        int print_count{0};
        for(int c=0; c<shape[1]; c++){
            std::cout<<'['<<std::endl;
            for(int i = 0; i < shape[2]; i++)
            {
                for(int j = 0; j < shape[3]; j++)
                {
                    if(j == 0)
                        std::cout<<"[";
                    std::cout<<batchData[print_count]<<",";
                    if(j == shape[3]-1)
                        std::cout<<"],";
                    print_count++;
                }
                std::cout<<std::endl;
            }
            std::cout<<"],"<<std::endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cout<<__FILE__<<":"<<__LINE__<<std::endl;std::cerr << e.what() << '\n';
    }
        }
    catch(const std::exception& e)
    {
        std::cout<<__FILE__<<":"<<__LINE__<<std::endl;std::cerr << e.what() << '\n';
    }
        
    std::cout << "Hello, OpenVINO!" << std::endl;
    return 0;
}