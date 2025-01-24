#include <iostream>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// clang-format off
#include <openvino/openvino.hpp>
#include "openvino/openvino.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/core/type/element_type.hpp"

class Weights{
    public:
    void const* values;
    int64_t count;
};
std::shared_ptr<ov::opset15::BatchNormInference> addBatchNorm2d(ov::Output<ov::Node>& input, std::map<std::string, Weights>& weights, std::string name, float eps=1e-5);
std::shared_ptr<ov::opset15::Multiply> convBnSiLu(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, uint64_t ch, uint64_t k, uint64_t s, int64_t p, std::string name);
std::shared_ptr<ov::op::Op> bottleneck(ov::Output<ov::Node> &input, std::map<std::string, Weights> &weights, uint64_t c1, uint64_t c2, bool shortcut, float e, std::string name);
std::shared_ptr<ov::opset15::Multiply> C2F(ov::Output<ov::Node>& input, std::map<std::string, Weights> &weights, uint64_t c1, uint64_t c2, uint64_t n, bool shortcut, float e, std::string name);
std::shared_ptr<ov::opset15::Multiply> SPPF(ov::Output<ov::Node>& input, std::map<std::string, Weights> &weights, uint64_t c1, uint64_t c2, uint64_t k, std::string name);
std::shared_ptr<ov::opset15::Interpolate> upsample(ov::Output<ov::Node> &input, std::vector<float> &scales);
std::shared_ptr<ov::op::Op> DFL(ov::Output<ov::Node> &inputs, std::map<std::string, Weights> &weights, uint64_t ch1, std::string name);
std::shared_ptr<ov::op::Op> YOLODetection(std::vector<ov::Output<ov::Node>> &inputs, std::map<std::string, Weights> &weights, std::string name);