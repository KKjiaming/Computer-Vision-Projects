//
// Created by kun on 2022/4/28.
//

#include "Yolov5_Trt.h"

using namespace smartmore;

Yolov5_Trt::Yolov5_Trt() = default;

Yolov5_Trt::~Yolov5_Trt() = default;

ResultCode Yolov5_Trt::Init(const std::vector<unsigned char> &model_data,
                            const std::string &model_name,
                            const cv::Size &input_size,
                            int num_classes, bool auto_size,
                            bool use_gpu, bool useFP16, int device_id) {

    // ------------------------ Infer Engine Config ------------------------
    smartmore::InitConfig detectInitConfig;
    detectInitConfig.modelData = model_data;
    detectInitConfig.gpuID = device_id;
    detectInitConfig.modelCachePath = model_name + ".cache";
    detectInitConfig.useFP16 = useFP16;

    // ------------------------- Init Infer Engine -------------------------
    mInferEngine = std::make_shared<smartmore::TRTModule>();
    mInferEngine->init(detectInitConfig);

    return Yolov5::Init(input_size, num_classes, auto_size);
}

ResultCode Yolov5_Trt::run(const cv::Mat &image,
                           float score_thresh,
                           float nms_thresh,
                           std::vector<DetObject> &results) {
    // ---------------------------- Get Input Info ----------------------------
    TensorInfo inputTensor = mInferEngine->getInputTensor("images");
    auto *inputHostBuffer = static_cast<float *>(inputTensor.hostBuffer);

    img_preprocess(image, inputHostBuffer);

    // ------------------------------- Inference -------------------------------
    mInferEngine->syncInput();
    mInferEngine->run();
    mInferEngine->syncOutput();
    // ----------------------------- Post Process -----------------------------
    smartmore::TensorInfo inferOutputTensor = mInferEngine->getOutputTensor("output");
    postprocess(inferOutputTensor, score_thresh, nms_thresh, results);

    return ResultCode::Success;
}

void Yolov5_Trt::postprocess(const smartmore::TensorInfo& inferOutputTensor,
                             float score_thresh,
                             float nms_thresh,
                             std::vector<DetObject> &results) {

    auto* inferOutputData = reinterpret_cast<float*>(inferOutputTensor.hostBuffer);

    assert(inferOutputTensor.dimInfo.size() == 3);

    auto inferOutputC = static_cast<int>(inferOutputTensor.dimInfo[0]);
    auto inferOutputH = static_cast<int>(inferOutputTensor.dimInfo[1]);
    auto inferOutputW = static_cast<int>(inferOutputTensor.dimInfo[2]);

    assert(inferOutputC == 1);
    assert(inferOutputW == (num_classes_+5));
    int num_anchors = inferOutputH;
    parseOutput(inferOutputData, results, num_anchors, score_thresh, nms_thresh, false);

}
