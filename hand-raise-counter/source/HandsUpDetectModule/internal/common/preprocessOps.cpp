//
// Created by kun on 2022/4/8.
//

#include "preprocessOps.h"

namespace smartmore {

    void ResizeImgType0(const cv::Mat &img, cv::Mat &resize_img, int max_size_len) {
        int w = img.cols;
        int h = img.rows;
        float ratio = 1.0;
        if (std::max(w, h) > max_size_len) {
            ratio = h > w ? float(max_size_len) / float(h) : float(max_size_len) / float(w);
        }
        int resize_h = int(float(h) * ratio);
        int resize_w = int(float(w) * ratio);
        resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);
        resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
        cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    }

    void ResizeImgType1(const cv::Mat &img, cv::Mat &resize_img, int max_long_size, int max_short_size) {
        int h = img.rows, w = img.cols;
        auto max_wh = float(std::max(h, w));
        float scale_factor = std::min(float(max_long_size) / max_wh, float(max_short_size) / max_wh);
        int resize_w = int(round(float(w) * scale_factor));
        int resize_h = int(round(float(h) * scale_factor));
        cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    }

    void Normalize(cv::Mat &img, const std::vector<float> &mean, const std::vector<float> &stdDev, bool is_scale) {
        assert(img.channels() == mean.size() && mean.size() == stdDev.size());
        double factor = 1.0;
        if (is_scale) factor /= 255.0;
        std::vector<cv::Mat> plates;
        cv::split(img, plates);
        for (size_t i = 0; i < plates.size(); ++i) {
            plates[i].convertTo(plates[i], CV_32FC1, factor);
            plates[i] = (plates[i] - mean[i]) / stdDev[i];
        }
        cv::merge(plates, img);
    }

    void BGR2RGB(cv::Mat &img) {
        assert(img.channels() == 3);
        std::vector<cv::Mat> plates;
        cv::split(img, plates);
        std::reverse(plates.begin(), plates.end());
        cv::merge(plates, img);
    }

    void PaddingImg(cv::Mat &img, int divisor) {
        int h = img.rows, w = img.cols;
        auto _div = float(divisor);
        int pad_h = int(round(float(h) / _div)) * divisor;
        int pad_w = int(round(float(w) / _div)) * divisor;
        cv::copyMakeBorder(img, img, 0, pad_h - h, 0, pad_w - w, cv::BORDER_CONSTANT, 0);
    }

    void CvMatHWC2CHW(cv::Mat &img, float* data) {
        img.convertTo(img, CV_32FC(img.channels()));
        // HWC to CHW
        int H = img.rows, W = img.cols, C = img.channels();
        for (int i = 0; i < H; ++i) {
            auto ptr = img.ptr<float>(i);
            for (int j = 0; j < W; ++j) {
                for (int k = 0; k < C; ++k) {
                    data[k * W * H + i * W + j] = *ptr;
                    ++ptr;
                }
            }
        }
    }

    std::vector<float> Permute(const cv::Mat &img) {
        int h = img.rows, w = img.cols, c = img.channels();
        std::vector<float> res(h * w * c);
        for (int i = 0; i < h; ++i) {
            auto ptr = img.ptr<uchar>(i);
            for (int j = 0; j < w; ++j) {
                for (int k = 0; k < c; ++k) {
                    res[k * w * h + i * w + j] = static_cast<float>(*ptr);
                    ++ptr;
                }
            }
        }
        return res;
    }

    std::vector<float> PermuteBatch(const std::vector<cv::Mat> &img) {
        int n = (int) img.size(), h = img[0].rows, w = img[0].cols, c = img[0].channels();
        std::vector<float> res(n * h * w * c);
        for (int idx = 0; idx < n; ++idx) {
            for (int i = 0; i < h; ++i) {
                auto ptr = img[idx].ptr<uchar>(i);
                for (int j = 0; j < w; ++j) {
                    for (int k = 0; k < c; ++k) {
                        res[idx * w * h * c + k * w * h + i * w + j] = static_cast<float>(*ptr);
                        ++ptr;
                    }
                }
            }
        }
        return res;
    }

}