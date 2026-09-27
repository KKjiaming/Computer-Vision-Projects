//
// Created by kun on 2022/4/11.
//
#include "commonFun.h"
#include <memory>
#include <fstream>

namespace smartmore {

    std::string getModelData(const std::string& model_path) {

        auto fs = std::make_shared<std::fstream>();
        fs->open(model_path, std::ios::binary | std::ios::in);
        if(!fs->is_open()){
            char info[512];
            sprintf(info, "read model file %s failed !\n", model_path.c_str());
            throw std::runtime_error(info);
        }
        fs->seekg(0, std::ios::end);
        long length = fs->tellg();
        fs->seekg(0, std::ios::beg);
        std::string modelData(length, ' ');
        fs->read(modelData.data(), length);
        fs->close();

        return modelData;
    }

    std::string stripPostfix(const std::string &str) {
        int idx = str.find_last_of('.');
        return str.substr(0, idx);
    }

    void crop_img(const cv::Mat &image, cv::Mat &_img, const BBox& box) {
        int h = image.rows, w = image.cols;
        int xmin = static_cast<int>(box.xmin);
        int ymin = static_cast<int>(box.ymin);
        int xmax = static_cast<int>(box.xmax);
        int ymax = static_cast<int>(box.ymax);
        xmin = std::max(0, xmin);
        ymin = std::max(0, ymin);
        xmax = std::min(w, xmax);
        ymax = std::min(h, ymax);
        if (xmin >= xmax || ymin >= ymax)
            return;
        image(cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin)).copyTo(_img);
    }

    void filter_roi_res(std::vector<DetObject> &objects, const cv::Rect& roi) {
        std::vector<DetObject> results;
        auto left = (float)roi.x;
        auto top = (float)roi.y;
        auto right = left + (float)roi.width;
        auto bottom = top + (float)roi.height;
        for (const auto& obj : objects) {
            auto x0 = (obj.bbox.xmin + obj.bbox.xmax) * 0.5f;
            auto y0 = (obj.bbox.ymin + obj.bbox.ymax) * 0.5f;
            if (x0 <= left || x0 >= right || y0 <= top || y0 >= bottom)
                continue;
            results.emplace_back(obj);
        }
        objects.swap(results);
    }

    bool check_valid_roi(const cv::Rect &roi, int imgHeight, int imgWidth) {
        return roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= imgWidth && roi.y + roi.height <= imgHeight;
    }

}