//
// Created by kun on 2022/4/20.
//

#ifndef SHIPDETECTOCR_Yolov5_H
#define SHIPDETECTOCR_Yolov5_H

#include "opencv2/opencv.hpp"
#include "../common/data_types.h"
#include <iostream>

namespace smartmore {
    class Yolov5 {
    public:
        Yolov5();

        ~Yolov5();

    public:
        ResultCode Init(const cv::Size &input_size, int num_classes, bool auto_size);

    protected:
        void img_preprocess(const cv::Mat &image, cv::Mat &_img);
        void img_preprocess(const cv::Mat &image, float *data);
        void parseOutput(const float* data,
                         std::vector<DetObject> &results,
                         int num_anchors,
                         float conf_thres,
                         float iou_thres,
                         bool agnostic_nms);

    private:
        void non_max_suppression(std::vector<DetObject> &detObjects,
                                 std::vector<DetObject> &results,
                                 float nms_thresh, bool agnostic_nms) const;

    protected:
        int num_classes_{};
        int input_h_{};
        int input_w_{};
        int num_anchors_{};

    private:
        bool auto_size_{};
        int stride_ = 32;
        int max_wh_ = 7680;
        int max_nms_ = 30000;
        cv::Size input_size_;

        float gain_{};
        std::pair<float, float> pad_;
        std::pair<float, float> img_size_;

    };
}

#endif //SHIPDETECTOCR_Yolov5_H
