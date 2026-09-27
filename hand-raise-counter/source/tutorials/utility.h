//
// Created by kun on 2022/4/13.
//

#ifndef HandsUpVOTINGDETECT_UTILITY_H
#define HandsUpVOTINGDETECT_UTILITY_H

#include "HandsUpDetectModule.h"
#include "opencv2/opencv.hpp"
#include "Common.h"
#include <chrono>
#ifdef _WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

typedef std::chrono::duration<double, std::ratio<1, 1000>> sm_ms;
typedef std::chrono::duration<double, std::ratio<1, 1>> sm_s;

namespace smartmore {

    void visualizeRoiArea(cv::Mat &visualImage,
                          const cv::Rect& roi,
                          const cv::Scalar& color);

    void visualizeAllPoses(cv::Mat &image, const std::vector<HandsUpResult> &HandsUpResults);

    void visualizeHandsUpResults(cv::Mat &image, const std::vector<HandsUpResult> &HandsUpResults, int handsUpNums, long long frame_idx);

}


#endif //HandsUpVOTINGDETECT_UTILITY_H
