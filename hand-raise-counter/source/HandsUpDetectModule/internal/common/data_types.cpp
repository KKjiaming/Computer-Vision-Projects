//
// Created by kun on 2022/4/8.
//
#include "data_types.h"

using namespace smartmore;

void DetObject::clip_coord(float h, float w) {
    bbox.xmin = std::min(w, std::max(bbox.xmin, 0.f));
    bbox.xmax = std::min(w, std::max(bbox.xmax, 0.f));
    bbox.ymin = std::min(h, std::max(bbox.ymin, 0.f));
    bbox.ymax = std::min(h, std::max(bbox.ymax, 0.f));
}

int64_t NodeShape::Size() const  {
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

int64_t TensorNode::size() const {
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

float TensorNode::index(const std::vector<int> &idxs) const {
    assert(idxs.size() == shape.size());
    for (size_t i = 0; i < idxs.size(); ++i)
        assert(idxs[i] >= 0 && idxs[i] < shape[i]);
    auto size_arr = [](const std::vector<int64_t> &x) -> std::vector<int64_t> {
        std::vector<int64_t> ans(x.size(), 1);
        for (int i = (int) x.size() - 2; i >= 0; --i)
            ans[i] = ans[i + 1] * x[i + 1];
        return ans;
    }(shape);
    int64_t idx = 0;
    for (size_t i = 0; i < idxs.size(); ++i) {
        idx += idxs[i] * size_arr[i];
    }
    return data[idx];
}


Tensor::Tensor(const float *data, std::vector<int64_t> &shape) {
    tensorShape = shape;
    auto size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    tensorMemory.resize(size);
    memcpy(tensorMemory.data(), data, size * 4);
}

Tensor::Tensor(std::vector<float> data, std::vector<int64_t> shape) :
        tensorMemory(std::move(data)), tensorShape(std::move(shape)) {

}

bool Tensor::isEmpty() const {
    return tensorMemory.empty();
}

std::vector<int64_t> Tensor::Shape() const {
    return tensorShape;
}

float Tensor::at(const std::vector<int> &idxs) const {
    assert(idxs.size() == tensorShape.size());
    for (size_t i = 0; i < idxs.size(); ++i)
        assert(idxs[i] >= 0 && idxs[i] < tensorShape[i]);
    auto size_arr = [](const std::vector<int64_t> &x) -> std::vector<int64_t> {
        std::vector<int64_t> ans(x.size(), 1);
        for (int i = (int) x.size() - 2; i >= 0; --i)
            ans[i] = ans[i + 1] * x[i + 1];
        return ans;
    }(tensorShape);
    int64_t idx = 0;
    for (size_t i = 0; i < idxs.size(); ++i) {
        idx += idxs[i] * size_arr[i];
    }
    return tensorMemory[idx];
}

int64_t Tensor::Size() const {
    return std::accumulate(tensorShape.begin(), tensorShape.end(), 1LL, std::multiplies<int64_t>());
}

const float *Tensor::data() {
    return tensorMemory.data();
}

float *Tensor::mutable_data() {
    return tensorMemory.data();
}

Tensor Tensor::fromCVMat(cv::Mat &img) {
    if (img.channels() == 1)
        img.convertTo(img, CV_32FC1);
    else if (img.channels() == 3)
        img.convertTo(img, CV_32FC3);
    else
        throw std::runtime_error("image channel must be 1 or 3!");
    int64_t h = img.rows, w = img.cols, c = img.channels();

    std::vector<float> data(h * w * c);
#if 1
    int nr = h;
    int nc = w;
    if (img.isContinuous()) {
        nr = 1;
        nc = nc * h * c;
    }
    int internal = nc / 3;
    int internalX2 = nc / 3 * 2;
    for (int k = 0; k < nr; k++) {
        auto pImage = img.ptr<float>(k);
        for (int cc = 0, idx = 0; cc < nc; cc += 3, ++idx) {
            data[idx] = *pImage++;
            data[idx + internal] = *pImage++;
            data[idx + internalX2] = *pImage++;
        }
    }
#else
    for (int i = 0; i < h; ++i) {
        auto ptr = img.ptr<float>(i);
        for (int j = 0; j < w; ++j) {
            for (int k = 0; k < c; ++k) {
                data[k * w * h + i * w + j] = *ptr;
                ++ptr;
            }
        }
    }
#endif
    return Tensor(data, {1, c, h, w});
}

Tensor Tensor::fromCVMats(std::vector<cv::Mat> &imgs) {
    assert(!imgs.empty());
    int c = imgs[0].channels();
    int n = static_cast<int>(imgs.size());
    int h = imgs[0].rows;
    int w = imgs[0].cols;
    std::vector<float> data(n * h * w * c);
    for (int idx = 0; idx < n; ++idx) {
        if (imgs[idx].channels() == 1)
            imgs[idx].convertTo(imgs[idx], CV_32FC1);
        else if (imgs[idx].channels() == 3)
            imgs[idx].convertTo(imgs[idx], CV_32FC3);
        else
            throw std::runtime_error("image channel must be 1 or 3!");
        for (int i = 0; i < h; ++i) {
            auto ptr = imgs[idx].ptr<float>(i);
            for (int j = 0; j < w; ++j) {
                for (int k = 0; k < c; ++k) {
                    data[idx * w * h * c + k * w * h + i * w + j] = *ptr;
                    ++ptr;
                }
            }
        }
    }
    return Tensor(data, {n, c, h, w});
}
