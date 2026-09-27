//
// Created by kun on 2022/4/28.
//

#include "Yolov5_Ort.h"
#include "../common/data_types.h"
#include <fstream>

using namespace smartmore;

Yolov5_Ort::Yolov5_Ort() = default;

Yolov5_Ort::~Yolov5_Ort() = default;

ResultCode Yolov5_Ort::Init(const std::vector<unsigned char> &model_data,
                            const std::string &model_name,
                            const cv::Size &input_size,
                            int num_classes, bool auto_size,
                            bool use_gpu, int device_id) {

    auto ret = OrtInferEngine::Init(model_data, use_gpu, device_id);
    if (ret != ResultCode::Success)
        return ResultCode::NotInited;
    return Yolov5::Init(input_size, num_classes, auto_size);
}

ResultCode Yolov5_Ort::run(const cv::Mat &image,
                           float conf_thresh,
                           float iou_thresh,
                           std::vector<DetObject> &results) {
    cv::Mat _img;
    img_preprocess(image, _img);
    std::vector<Tensor> in_data{Tensor::fromCVMat(_img)};

    // std::ofstream fout("input.txt");
    // assert (fout.is_open());
    // for (int i = 0; i < 3 * 640 * 640; ++i)
    //     fout << std::setprecision(30) << inputHostBuffer[i] << " ";
    // fout << "\n";
    // fout.close();

    // ------------------------------- Inference -------------------------------
    // auto t1 = cv::getTickCount();
    std::vector<TensorNode> out_data;
    OrtInferEngine::Inference(in_data, out_data);
    // printf("model infer %f\n", (double)(cv::getTickCount()-t1)/cv::getTickFrequency());
    // ----------------------------- Post Process -----------------------------
    postprocess(out_data[0], conf_thresh, iou_thresh, results);

    return ResultCode::Success;
}

void Yolov5_Ort::postprocess(const TensorNode& outNode,
                             float conf_thresh,
                             float iou_thresh,
                             std::vector<DetObject> &results) {


    assert(outNode.shape.size() == 3);

    // std::ofstream fout("output.txt");
    // assert (fout.is_open());
    // for (int i = 0; i < inferOutputC * inferOutputH * inferOutputW; ++i)
    //     fout << std::setprecision(30) << inferOutputData[i] << " ";
    // fout << "\n";
    // fout.close();

    assert(outNode.shape[0] == 1);
    assert(outNode.shape[2] == (num_classes_+5));
    num_anchors_ = static_cast<int>(outNode.shape[1]);
    parseOutput(outNode.data, results, num_anchors_, conf_thresh, iou_thresh, false);

}
