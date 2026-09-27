//
// Created by kun on 2022/4/28.
//

#ifndef YOLOV5_TRT_H
#define YOLOV5_TRT_H

#include "Yolov5.h"
#include "InferModule/TRTModule.h"

namespace smartmore {

    class Yolov5_Trt : public Yolov5 {
    public:
        Yolov5_Trt();

        ~Yolov5_Trt();

        ResultCode Init(const std::vector<unsigned char> &model_data,
                        const std::string &model_name,
                        const cv::Size &input_size,
                        int num_classes, bool auto_size,
                        bool use_gpu, bool useFP16, int device_id);

        ResultCode run(const cv::Mat &image,
                       float score_thresh,
                       float nms_thresh,
                       std::vector<DetObject> &results);

    private:

        void postprocess(const smartmore::TensorInfo& inferOutputTensor,
                         float score_thresh,
                         float nms_thresh,
                         std::vector<DetObject> &results);

    private:
        std::shared_ptr<TRTModule> mInferEngine;
    };

}

#endif //SMARTCITYPROJECT_YOLOV5_TRT_H
