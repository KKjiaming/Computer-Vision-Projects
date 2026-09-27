//
// Created by kun on 2022/4/8.
//

#ifndef VIDEOANALYSIS_PREPROCESSOPS_H
#define VIDEOANALYSIS_PREPROCESSOPS_H

#include <cmath>
#include "opencv2/opencv.hpp"

namespace smartmore {

    void ResizeImgType0(const cv::Mat &img, cv::Mat &resize_img, int max_size_len);

    void ResizeImgType1(const cv::Mat &img, cv::Mat &resize_img, int max_long_size, int max_short_size);

    void Normalize(cv::Mat &img, const std::vector<float> &mean, const std::vector<float> &stdDev, bool is_scale);

    void BGR2RGB(cv::Mat &img);

    void PaddingImg(cv::Mat &img, int divisor);

    void CvMatHWC2CHW(cv::Mat &img, float* data);

    std::vector<float> Permute(const cv::Mat &img);

    std::vector<float> PermuteBatch(const std::vector<cv::Mat> &img);

}

#endif //VIDEOANALYSIS_PREPROCESSOPS_H
