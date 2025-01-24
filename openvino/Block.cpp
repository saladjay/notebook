#include "Block.h"

std::shared_ptr<ov::opset15::BatchNormInference> addBatchNorm2d(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, std::string name, float eps)
{
    using namespace ov;
    using namespace ov::opset15;
    // get nb of channels
    auto input_c = static_cast<uint64_t>(input.get_partial_shape()[1].get_length());
    // Create constants for scale, bias, mean, var

    auto bnScale = std::make_shared<Constant>(element::Type_t::f32, Shape{input_c}, weights[name + ".weight"].values);

    auto bnB = std::make_shared<Constant>(element::Type_t::f32, Shape{input_c}, weights[name + ".bias"].values);

    auto bnMean = std::make_shared<Constant>(element::Type_t::f32, Shape{input_c}, weights[name + ".running_mean"].values);

    auto bnVar = std::make_shared<Constant>(element::Type_t::f32, Shape{input_c}, weights[name + ".running_var"].values);

    // (input - running_mean) / sqrt(running_var + eps) * weight + bias

    auto bn = std::make_shared<BatchNormInference>(input, bnScale->output(0), bnB->output(0), bnMean->output(0), bnVar->output(0), eps);
    return bn;
}

std::shared_ptr<ov::opset15::Multiply> convBnSiLu(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, uint64_t ch, uint64_t k, uint64_t s, int64_t p, std::string name)
{
    using namespace ov;
    using namespace ov::opset15;
    auto input_c = static_cast<uint64_t>(input.get_partial_shape()[1].get_length());

    auto convFirstShape = ov::Shape{ch, input_c, k, k}; // output_c, input_c, kernel_h, kernel_w

    auto convolutionFilter = std::make_shared<Constant>(element::Type_t::f32, convFirstShape, weights[name + ".conv.weight"].values);

    // Create a vector of input shapes
    std::vector<ptrdiff_t> padBegin{p, p}; // top, left
    std::vector<ptrdiff_t> padEnd{p, p};   // bottom, right

    auto conv = std::make_shared<Convolution>(input,
                                              convolutionFilter->output(0),
                                              Strides({1, 1}),
                                              CoordinateDiff(padBegin),
                                              CoordinateDiff(padEnd),
                                              Strides({1, 1}));

    std::shared_ptr<op::Op> layer = conv;
    // bias
    if (weights.find(name + ".conv.bias") != weights.end())
    {
        auto biasShape = Shape{1, ch, 1, 1};

        auto biasValues = std::make_shared<Constant>(element::Type_t::f32, biasShape, weights[name + ".conv.bias"].values);
        auto addNode = std::make_shared<Add>(conv->output(0), biasValues->output(0));
        layer = addNode;
    }

    // BatchNormalization
    auto bn = addBatchNorm2d(layer->output(0), weights, name + ".bn", 1e-3f);

    // SiLU
    auto sigmoid = std::make_shared<Sigmoid>(bn->output(0));
    auto mulNode = std::make_shared<Multiply>(bn->output(0), sigmoid->output(0));

    return mulNode;
}

std::shared_ptr<ov::op::Op> bottleneck(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, uint64_t c1, uint64_t c2, bool shortcut, float e, std::string name)
{
    auto conv1 = convBnSiLu(input, weights, c2, 3, 1, 1, name + ".cv1");
    auto conv2 = convBnSiLu(conv1->output(0), weights, c2, 3, 1, 1, name + ".cv2");
    if (shortcut && c1 == c2)
    {
        auto add = std::make_shared<ov::opset15::Add>(input, conv2->output(0));
        return add;
    }
    return conv2;
}

std::shared_ptr<ov::opset15::Multiply> C2F(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, uint64_t c1, uint64_t c2, uint64_t n, bool shortcut, float e, std::string name)
{
    using namespace ov;
    using namespace ov::opset15;

    uint64_t c_ = static_cast<uint64_t>((float)c2 * e);
    auto conv1 = convBnSiLu(input, weights, 2 * c_, 1, 1, 0, name + ".cv1");
    auto shape = conv1->get_output_shape(0);

    int b = static_cast<int>(shape[0]);
    int c = static_cast<int>(shape[1]);
    int h = static_cast<int>(shape[2]);
    int w = static_cast<int>(shape[3]);
    // Create parameters for Slice operation
    auto split1 = std::make_shared<Slice>(conv1->output(0),
                                          Constant::create(ov::element::u64, {4}, {0, 0, 0, 0}),
                                          Constant::create(ov::element::u64, {4}, {b, c / 2, h, w}),
                                          Constant::create(ov::element::u64, {4}, {1, 1, 1, 1}));

    auto split2 = std::make_shared<Slice>(conv1->output(0),
                                          Constant::create(ov::element::u64, {4}, {0, c / 2, 0, 0}),
                                          Constant::create(ov::element::u64, {4}, {b, c, h, w}),
                                          Constant::create(ov::element::u64, {4}, {1, 1, 1, 1}));

    auto cat = std::make_shared<Concat>(std::vector<Output<Node>>{split1->output(0), split2->output(0)}, 1);

    auto y1 = split2->output(0);
    for (uint64_t i = 0; i < n; i++)
    {
        auto b = bottleneck(y1, weights, c_, c_, shortcut, 1.0, name + ".m." + std::to_string(i));
        y1 = b->output(0);
        cat = std::make_shared<Concat>(std::vector<Output<Node>>{cat->output(0), y1}, 1);
    }

    auto conv2 = convBnSiLu(cat->output(0), weights, c2, 1, 1, 0, name + ".cv2");

    return conv2;
}

std::shared_ptr<ov::opset15::Multiply> SPPF(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, uint64_t c1, uint64_t c2, uint64_t k, std::string name)
{
    using namespace ov;
    using namespace ov::opset15;
    auto c_ = c1 / 2;
    auto conv1 = convBnSiLu(input, weights, c_, 1, 1, 0, name + ".cv1");
    auto pool1 = std::make_shared<MaxPool>(conv1->output(0), Strides({1, 1}), Strides({1, 1}),
                                           Shape({k / 2, k / 2}), Shape({k / 2, k / 2}), Shape({k, k}), op::RoundingType::CEIL_TORCH, op::PadType::EXPLICIT);
    auto pool2 = std::make_shared<MaxPool>(pool1->output(0), Strides({1, 1}), Strides({1, 1}),
                                           Shape({k / 2, k / 2}), Shape({k / 2, k / 2}), Shape({k, k}), op::RoundingType::CEIL_TORCH, op::PadType::EXPLICIT);
    auto pool3 = std::make_shared<MaxPool>(pool2->output(0), Strides({1, 1}), Strides({1, 1}),
                                           Shape({k / 2, k / 2}), Shape({k / 2, k / 2}), Shape({k, k}), op::RoundingType::CEIL_TORCH, op::PadType::EXPLICIT);
    auto cat = std::make_shared<Concat>(std::vector<ov::Output<ov::Node>>{conv1->output(0), pool1->output(0), pool2->output(0), pool3->output(0)}, 1);
    auto conv2 = convBnSiLu(cat->output(0), weights, c2, 1, 1, 0, name + ".cv2");
    return conv2;
}

std::shared_ptr<ov::opset15::Interpolate> upsample(ov::Output<ov::Node> &input, std::vector<float> &scales)
{
    using namespace ov;
    using namespace ov::opset15;
    op::util::InterpolateBase::InterpolateAttrs attr;
    attr.mode = op::v4::Interpolate::InterpolateMode::NEAREST;
    attr.shape_calculation_mode = op::v4::Interpolate::ShapeCalcMode::SCALES;
    attr.nearest_mode = op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    auto resize = std::make_shared<Interpolate>(input, Constant::create(ov::element::f32, {4}, scales), attr);
    return resize;

    // std::make_shared<Reshape>
}

std::shared_ptr<ov::op::Op> DFL(ov::Output<ov::Node> &inputs, std::map<std::string, Weights> &weights, uint64_t ch1, std::string name)
{
    using namespace ov;
    using namespace ov::opset15;

    auto input_shape = inputs.get_shape();
    auto batch = input_shape[0];
    auto channel = input_shape[1];
    auto anchor = input_shape[2];

    auto reshape = std::make_shared<Reshape>(inputs, Constant::create(ov::element::i64, {4}, {-1, 4, 16, 3}), false);
    auto transpose = std::make_shared<Transpose>(reshape->output(0), Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3}));
    auto softmax = std::make_shared<Softmax>(transpose->output(0), 1);

    auto input_c = static_cast<uint64_t>(softmax->output(0).get_partial_shape()[1].get_length());
    auto convFirstShape = ov::Shape{ch1, input_c, 1, 1}; // output_c, input_c, kernel_h, kernel_w
    auto convolutionFilter = std::make_shared<Constant>(element::Type_t::f32, convFirstShape, weights[name + ".conv.weight"].values);

    // Create a vector of input shapes
    std::vector<ptrdiff_t> padBegin{0, 0}; // top, left
    std::vector<ptrdiff_t> padEnd{0, 0};   // bottom, right

    auto conv = std::make_shared<Convolution>(softmax->output(0),
                                              convolutionFilter->output(0),
                                              Strides({1, 1}),
                                              CoordinateDiff(padBegin),
                                              CoordinateDiff(padEnd),
                                              Strides({1, 1}));

    auto reshape2 = std::make_shared<Reshape>(conv->output(0), Constant::create(ov::element::i64, {3}, {-1, 4, static_cast<int>(anchor)}), false);
    return reshape2;
}

std::shared_ptr<ov::op::Op> YOLODetection(std::vector<ov::Output<ov::Node>> &inputs, std::map<std::string, Weights> &weights, std::string name)
{
    using namespace ov;
    using namespace ov::opset15;
    if (inputs.size() != 3)
    {
        std::cout << "YOLODetection: inputs.size() != 3" << std::endl;
        return std::shared_ptr<ov::op::Op>();
    }
    auto shape = inputs[0].get_shape();
    int batch = static_cast<int>(shape[0]);
    std::vector<int> strides{8, 16, 32};
    int input_width = 640;
    int input_height = 640;
    int nb_categories = 80;
    // stride 8
    auto shape_layer1 = std::make_shared<Reshape>(inputs[0], Constant::create(ov::element::i64, {3}, {-1, 64 + nb_categories, input_height / strides[0] * input_width / strides[0]}), false);
    auto split1_layer1 = std::make_shared<Slice>(shape_layer1->output(0),
                                                 Constant::create(ov::element::u64, {3}, {0, 0, 0}),
                                                 Constant::create(ov::element::u64, {3}, {batch, 64, input_height / strides[0] * input_width / strides[0]}),
                                                 Constant::create(ov::element::u64, {3}, {1, 1, 1}));
    auto split2_layer1 = std::make_shared<Slice>(shape_layer1->output(0),
                                                 Constant::create(ov::element::u64, {3}, {0, 64, 0}),
                                                 Constant::create(ov::element::u64, {3}, {batch, 64 + nb_categories, input_height / strides[0] * input_width / strides[0]}),
                                                 Constant::create(ov::element::u64, {3}, {1, 1, 1}));
    auto cls_layer1 = std::make_shared<Sigmoid>(split1_layer1->output(0));    
    auto dfl_layer1 = DFL(split1_layer1->output(0), weights, 4, name + ".dfl");
    auto concat_layer1 = std::make_shared<Concat>(std::vector<Output<Node>>{dfl_layer1->output(0), cls_layer1->output(0)}, 1);

    // stride 16
    auto shape_layer2 = std::make_shared<Reshape>(inputs[1], Constant::create(ov::element::i64, {3}, {-1, 64 + nb_categories, input_height / strides[1] * input_width / strides[1]}), false);
    auto split1_layer2 = std::make_shared<Slice>(shape_layer2->output(0),
                                                 Constant::create(ov::element::u64, {3}, {0, 0, 0}),
                                                 Constant::create(ov::element::u64, {3}, {batch, 64, input_height / strides[1] * input_width / strides[1]}),
                                                 Constant::create(ov::element::u64, {3}, {1, 1, 1}));
    auto split2_layer2 = std::make_shared<Slice>(shape_layer2->output(0),
                                                 Constant::create(ov::element::u64, {3}, {0, 64, 0}),
                                                 Constant::create(ov::element::u64, {3}, {batch, 64 + nb_categories, input_height / strides[1] * input_width / strides[1]}),
                                                 Constant::create(ov::element::u64, {3}, {1, 1, 1}));
    auto cls_layer2 = std::make_shared<Sigmoid>(split1_layer2->output(0));                         
    auto dfl_layer2 = DFL(split1_layer2->output(0), weights, 4, name + ".dfl");
    auto concat_layer2 = std::make_shared<Concat>(std::vector<Output<Node>>{dfl_layer2->output(0), cls_layer2->output(0)}, 1);

    // stride 32
    auto shape_layer3 = std::make_shared<Reshape>(inputs[2], Constant::create(ov::element::i64, {3}, {-1, 64 + nb_categories, input_height / strides[2] * input_width / strides[2]}), false);
    auto split1_layer3 = std::make_shared<Slice>(shape_layer3->output(0),
                                                 Constant::create(ov::element::u64, {3}, {0, 0, 0}),
                                                 Constant::create(ov::element::u64, {3}, {batch, 64, input_height / strides[2] * input_width / strides[2]}),
                                                 Constant::create(ov::element::u64, {3}, {1, 1, 1}));
    auto split2_layer3 = std::make_shared<Slice>(shape_layer3->output(0),
                                                 Constant::create(ov::element::u64, {3}, {0, 64, 0}),
                                                 Constant::create(ov::element::u64, {3}, {batch, 64 + nb_categories, input_height / strides[2] * input_width / strides[2]}),
                                                 Constant::create(ov::element::u64, {3}, {1, 1, 1}));
    auto cls_layer3 = std::make_shared<Sigmoid>(split1_layer3->output(0));
    auto dfl_layer3 = DFL(split1_layer3->output(0), weights, 4, name + ".dfl");
    auto concat_layer3 = std::make_shared<Concat>(std::vector<Output<Node>>{dfl_layer3->output(0), cls_layer3->output(0)}, 1);
    auto concat_all = std::make_shared<Concat>(std::vector<Output<Node>>{concat_layer1->output(0), concat_layer2->output(0), concat_layer3->output(0)}, 2);
    return concat_all;
}
