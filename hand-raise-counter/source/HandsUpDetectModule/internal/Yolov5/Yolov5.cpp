//
// Created by kun on 2022/4/20.
//

#include "Yolov5.h"
#include "../common/postprocessOps.h"
#include "cmath"

using namespace smartmore;

Yolov5::Yolov5() = default;

Yolov5::~Yolov5() = default;

ResultCode Yolov5::Init(const cv::Size &input_size, int num_classes, bool auto_size) {
    input_size_ = input_size;
    num_classes_ = num_classes;
    auto_size_ = auto_size;
    return ResultCode::Success;
}

void Yolov5::img_preprocess(const cv::Mat &image, cv::Mat &_img) {
    auto h = (float) image.rows;
    auto w = (float) image.cols;
    img_size_ = std::make_pair(h, w);
    auto input_h = (float) input_size_.height;
    auto input_w = (float) input_size_.width;
    auto r = std::min(input_h / h, input_w / w);
    auto new_unpad_h = (int) (std::round(h * r));
    auto new_unpad_w = (int) (std::round(w * r));
    float dh, dw;
    if (auto_size_) {
        dh = static_cast<float>((input_size_.height - new_unpad_h) % stride_) / 2;
        dw = static_cast<float>((input_size_.width - new_unpad_w) % stride_) / 2;
    } else {
        dh = static_cast<float>(input_size_.height - new_unpad_h) / 2;
        dw = static_cast<float>(input_size_.width - new_unpad_w) / 2;
    }
    int top = static_cast<int>(round(dh - 0.1));
    int bottom = static_cast<int>(round(dh + 0.1));
    int left = static_cast<int>(round(dw - 0.1));
    int right = static_cast<int>(round(dw + 0.1));
    if (new_unpad_h != image.rows || new_unpad_w != image.cols)
        cv::resize(image, _img, cv::Size(new_unpad_w, new_unpad_h), cv::INTER_LINEAR);
    else
        image.copyTo(_img);
    cv::cvtColor(_img, _img, cv::COLOR_BGR2RGB);
    cv::copyMakeBorder(_img, _img,
                       top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar::all(114));
    _img.convertTo(_img, CV_32FC3, 1 / 255.0);
    input_h_ = static_cast<int>(_img.rows);
    input_w_ = static_cast<int>(_img.cols);
    gain_ = std::min(static_cast<float>(_img.rows) / h, static_cast<float>(_img.cols) / w);
    pad_ = std::make_pair((static_cast<float>(_img.cols) - w * gain_) / 2,
                          (static_cast<float>(_img.rows) - h * gain_) / 2);
}

void Yolov5::img_preprocess(const cv::Mat &image, float *data) {
    cv::Mat _img;
    img_preprocess(image, _img);

    // HWC to CHW
    int H = _img.rows, W = _img.cols, C = (int)_img.channels();
#if 1
    int nr = H;
    int nc = W;
    if (_img.isContinuous()) {
        nr = 1;
        nc = nc * H * C;
    }
    int internal = nc / 3;
    int internalX2 = nc / 3 * 2;
    for (int k = 0; k < nr; k++) {
        auto pImage = _img.ptr<float>(k);
        for (int c = 0, idx = 0; c < nc; c += 3, ++idx) {
            data[idx] = *pImage++;
            data[idx + internal] = *pImage++;
            data[idx + internalX2] = *pImage++;
        }
    }
#else
    for (int i = 0; i < H; ++i) {
        auto ptr = _img.ptr<float>(i);
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < C; ++k) {
                data[k * W * H + i * W + j] = *ptr;
                ++ptr;
            }
        }
    }
#endif
}

void Yolov5::parseOutput(const float *data,
                         std::vector<DetObject> &results,
                         int num_anchors,
                         float conf_thres,
                         float iou_thres,
                         bool agnostic_nms) {
    std::vector<DetObject> pre_results;
    pre_results.reserve(num_anchors * 2);
    for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
        int basic_pos = anchor_idx * (num_classes_ + 5);
        float obj_conf = data[basic_pos + 4];
        if (obj_conf <= conf_thres)
            continue;
        std::vector<float> cls_conf(num_classes_);
        float best_conf = std::numeric_limits<float>::min();
        int best_cls = -1;
        for (int i = 0; i < num_classes_; ++i) {
            cls_conf[i] = data[basic_pos + 5 + i] * obj_conf;
            if (cls_conf[i] > best_conf) {
                best_conf = cls_conf[i];
                best_cls = i;
            }
        }
        float x0 = data[basic_pos];
        float y0 = data[basic_pos + 1];
        float w = data[basic_pos + 2];
        float h = data[basic_pos + 3];
        if (best_conf > conf_thres) {
            BBox bbox{x0 - w / 2, y0 - h / 2, x0 + w / 2, y0 + h / 2, best_conf};
            pre_results.emplace_back(DetObject{best_cls, bbox});
        }
    }
    results.reserve(pre_results.size() * 2);
    non_max_suppression(pre_results, results, iou_thres, agnostic_nms);
    for (auto &obj : results) {
        obj.bbox.xmin = (obj.bbox.xmin - pad_.first) / gain_;
        obj.bbox.xmax = (obj.bbox.xmax - pad_.first) / gain_;
        obj.bbox.ymin = (obj.bbox.ymin - pad_.second) / gain_;
        obj.bbox.ymax = (obj.bbox.ymax - pad_.second) / gain_;
        obj.clip_coord(img_size_.first, img_size_.second);
    }
}


void Yolov5::non_max_suppression(std::vector<DetObject> &detObjects,
                                 std::vector<DetObject> &results,
                                 float nms_thresh, bool agnostic_nms) const {
    std::sort(detObjects.begin(), detObjects.end(), [](const DetObject &a, const DetObject &b) -> bool {
        return a.bbox.score > b.bbox.score;
    });
    if ((int)detObjects.size() > max_nms_)
        detObjects.resize(max_nms_);
    std::vector<DetObject> _objects(detObjects.begin(), detObjects.end());
    if (!agnostic_nms) {
        for (auto &obj : _objects) {
            auto offset = static_cast<float>(obj.cls_idx * max_wh_);
            obj.bbox.xmin += offset;
            obj.bbox.ymin += offset;
            obj.bbox.xmax += offset;
            obj.bbox.ymax += offset;
        }
    }
    std::vector<bool> keep(_objects.size(), true);
    for (size_t i = 0; i < _objects.size(); i++) {
        if (keep[i]) {
            for (size_t j = i + 1; j < _objects.size(); j++)
                keep[j] = keep[j] && bbox_iou(_objects[i].bbox, _objects[j].bbox) < nms_thresh;
            results.emplace_back(detObjects[i]);
        }
    }
}


