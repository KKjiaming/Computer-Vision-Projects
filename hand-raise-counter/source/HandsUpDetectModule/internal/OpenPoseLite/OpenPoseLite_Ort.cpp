//
// Created by kun on 2022/6/20.
//

#include "OpenPoseLite_Ort.h"

using namespace smartmore;


OpenPoseLite_Ort::OpenPoseLite_Ort() = default;

OpenPoseLite_Ort::~OpenPoseLite_Ort() = default;

ResultCode OpenPoseLite_Ort::Init(const std::vector<unsigned char> &model_data,
                                  const std::string &model_name,
                                  bool use_gpu, int device_id) {
    
    return OrtInferEngine::Init(model_data, use_gpu, device_id);
}

ResultCode OpenPoseLite_Ort::run(const cv::Mat &image, std::vector<Pose> &current_poses) {
    // auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat _img;
    img_preprocess(image, _img);
    std::vector<Tensor> in_data{Tensor::fromCVMat(_img)};
    std::vector<TensorNode> out_data;
    // auto t2 = std::chrono::high_resolution_clock::now();
    // printf("\033[32m[Demo] preprocess time: %.1f ms\n\033[0m",
    //         std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2-t1).count());
    // t1 = std::chrono::high_resolution_clock::now();
    OrtInferEngine::Inference(in_data, out_data);
    // t2 = std::chrono::high_resolution_clock::now();
    // printf("\033[32m[Demo] inference time: %.1f ms\n\033[0m",
    //         std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2-t1).count());
    // t1 = std::chrono::high_resolution_clock::now();
    
    // stage_1_output_1_heatmaps
    // stage_1_output_0_pafs

    postprocess(out_data[0], out_data[1], current_poses);
    // t2 = std::chrono::high_resolution_clock::now();
    // printf("\033[32m[Demo] postprocess time: %.1f ms\n\033[0m",
    //         std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2-t1).count());
    return ResultCode::Success;
}

void OpenPoseLite_Ort::postprocess(const TensorNode &heatMapsTensor,
                                   const TensorNode &pafsTensor,
                                   std::vector<Pose> &current_poses) {
    // assert(checkTensorShape(heatMapsTensor.shape, {1, 19, 64, 114}));
    // assert(checkTensorShape(pafsTensor.shape, {1, 38, 64, 114}));
    // if (!checkTensorShape(heatMapsTensor.shape, {1, 19, 64, 114}))
    //     throw std::runtime_error("shape not correct!");

    const float *heatMapsData = heatMapsTensor.data;
    auto heatMapsDims = heatMapsTensor.shape;
    int featureHeight = static_cast<int>(heatMapsDims[2]);
    int featureWidth = static_cast<int>(heatMapsDims[3]);
    int heatMapsOffset = featureHeight * featureWidth;
    int nHeatMaps = static_cast<int>(heatMapsDims[1]);

    const float *pafsData = pafsTensor.data;
    auto pafsDims = pafsTensor.shape;
    assert(heatMapsDims[2] == pafsDims[2]);
    assert(heatMapsDims[3] == pafsDims[3]);
    int pafsOffset = heatMapsOffset;
    int nPafs = static_cast<int>(pafsDims[1]);
    // auto t1 = std::chrono::high_resolution_clock::now();
    post_process(heatMapsData, pafsData,
                 heatMapsOffset, nHeatMaps, pafsOffset, nPafs,
                 featureHeight, featureWidth, 10, 0.36, current_poses);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // printf("\033[32m[Demo] postprocess time: %.1f ms\n\033[0m",
    //         std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2-t1).count());
}

bool OpenPoseLite_Ort::checkTensorShape(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2) {
    if (shape1.size() != shape2.size())
        return false;
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i])
            return false;
    }
    return true;
}
