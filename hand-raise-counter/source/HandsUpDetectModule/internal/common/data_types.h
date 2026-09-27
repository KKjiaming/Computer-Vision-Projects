//
// Created by kun on 2022/4/8.
//

#ifndef VIDEOANALYSIS_DATA_TYPES_H
#define VIDEOANALYSIS_DATA_TYPES_H

#include "opencv2/opencv.hpp"
#include "../../include/Common.h"
#include <iostream>
#include <vector>
#include <numeric>

namespace smartmore {

    struct DetObject {
        int cls_idx;
        BBox bbox;
        cv::Mat mask;
        void clip_coord(float h, float w);
    };

    class NodeShape {
    public:
        std::vector<int64_t> shape;
        int64_t Size() const;
    };

    struct TensorNode {
        float* data;
        std::vector<int64_t> shape;
        int64_t size() const;
        float index(const std::vector<int>& idxs) const;
        friend std::ostream &operator<<(std::ostream& os, const TensorNode& tensorNode);
    };

    class Tensor {
    private:
        std::vector<float> tensorMemory{};
        std::vector<int64_t> tensorShape{};

    public:
        Tensor() = default;

        Tensor(const float* data, std::vector<int64_t>& shape);

        Tensor(std::vector<float> data, std::vector<int64_t> shape);

        ~Tensor() = default;

        bool isEmpty() const;

        std::vector<int64_t> Shape() const;

        friend std::ostream &operator<<(std::ostream& os, const Tensor& tensor);

        float at(const std::vector<int>& idxs) const;

        int64_t Size() const;

        const float* data();

        float* mutable_data();

        static Tensor fromCVMat(cv::Mat& img);

        static Tensor fromCVMats(std::vector<cv::Mat>& imgs);

    };

}

#endif //VIDEOANALYSIS_DATA_TYPES_H
