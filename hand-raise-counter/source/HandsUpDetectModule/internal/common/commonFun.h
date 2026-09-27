//
// Created by kun on 2022/4/11.
//

#ifndef VIDEOANALYSIS_COMMONFUN_H
#define VIDEOANALYSIS_COMMONFUN_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include "Common.h"
#include "data_types.h"

namespace smartmore {

    std::string getModelData(const std::string& model_path);

    std::string stripPostfix(const std::string& str);

    void crop_img(const cv::Mat &image, cv::Mat &_img, const BBox& box);

    void filter_roi_res(std::vector<DetObject> &objects, const cv::Rect& roi);

    bool check_valid_roi(const cv::Rect &roi, int imgHeight, int imgWidth);

}

#endif //VIDEOANALYSIS_COMMONFUN_H
