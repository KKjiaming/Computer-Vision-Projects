//
// Created by kun on 2022/4/28.
//

#ifndef SHIPDETECTOCR_YOLOV5_ORT_H
#define SHIPDETECTOCR_YOLOV5_ORT_H

#include "Yolov5.h"
#include "../InferEngine/OrtInferEngine.h"

namespace smartmore {

    class Yolov5_Ort : public Yolov5, OrtInferEngine {
    public:
        Yolov5_Ort();

        ~Yolov5_Ort();

        ResultCode Init(const std::vector<unsigned char> &model_data,
                        const std::string &model_name,
                        const cv::Size &input_size,
                        int num_classes, bool auto_size,
                        bool use_gpu, int device_id);

        ResultCode run(const cv::Mat &image,
                       float conf_thresh,
                       float iou_thresh,
                       std::vector<DetObject> &results);

    private:

        void postprocess(const TensorNode& outNode,
                         float conf_thresh,
                         float iou_thresh,
                         std::vector<DetObject> &results);

    };

}

#endif //SHIPDETECTOCR_YOLOV5_ORT_H
