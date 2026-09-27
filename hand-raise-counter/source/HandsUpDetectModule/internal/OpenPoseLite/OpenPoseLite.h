//
// Created by kun on 2022/6/20.
//

#ifndef HandsUpVOTINGDETECT_OPENPOSELITE_H
#define HandsUpVOTINGDETECT_OPENPOSELITE_H

#include "opencv2/opencv.hpp"
#include <utility>
#include <vector>
#include "Pose.h"

namespace smartmore {

    struct Peak {
        int idx;
        float x, y, score;
        explicit Peak(int idx_ = -1, float x_ = -1, float y_ = -1, float score_ = 0.f):
            idx(idx_), x(x_), y(y_), score(score_) {}
        friend std::ostream& operator<<(std::ostream &os, const Peak &that) {
            os << that.x << " " << that.y << " " << that.score << " " << that.idx;
            return os;
        }
    };

    class OpenPoseLite {
    public:
        OpenPoseLite();

        ~OpenPoseLite();

    protected:
        void img_preprocess(const cv::Mat &image, cv::Mat &_img);

        void img_preprocess(const cv::Mat &image, float *data);

        void post_process(const float* heatMapsData,
                          const float* pafsData,
                          int heatMapOffset, int nHeatMaps,
                          int pafOffset, int nPafs,
                          int featureMapHeight, int featureMapWidth,
                          float conf_thresh, float keypoint_thresh,
                          std::vector<Pose> &current_poses);

    private:
        void pad_width(cv::Mat &img, int pad_value, std::vector<int> &min_dims);
        void resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const;
        static void extractKeyPoints(const std::vector<cv::Mat>& heatMaps,
                                     std::vector<std::vector<Peak>>& all_keyPoints_by_type,
                                     int heatMapIdx, float minPeakDistance = 6, float threshold = 0.1);
        void group_keyPoints(const std::vector<std::vector<Peak>> &all_keyPoint_by_type,
                             const std::vector<cv::Mat> &pafMaps,
                             std::vector<Peak> &all_keyPoints,
                             std::vector<std::vector<float>> &pose_entries,
                             float pose_entry_size = 20,
                             float min_paf_score = 0.05);
        void connection_nms(std::vector<std::tuple<int, int, float>> &affinity_idx_scores);

        bool keypointfilter(std::vector<float> &pose_entry, const std::vector<Peak> &all_keypoints, float keypoint_threshold);

    protected:
        int input_h_{};
        int input_w_{};
        int stride_ = 8;

    private:
        int height_size_ = 512;
        int upsample_ratio_ = 4;
        int num_keyPoints_ = 18;

        std::vector<int> pads_;
        float scale_{};

        std::vector<float> mean_{128.f, 128.f, 128.f};
        std::vector<float> stdDev_{256.f, 256.f, 256.f};

        const std::vector<std::pair<int, int>> BODY_PARTS_KPT_IDS_ = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
                                                                      {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};
        const std::vector<std::pair<int, int>> BODY_PARTS_PAF_IDS_ = {{12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1}, {2, 3}, {4, 5},
                                                                      {6, 7}, {8, 9}, {10, 11}, {28, 29}, {30, 31}, {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27}};
        const std::vector<int> UPPER_PARTS_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17};
        const std::vector<std::string> ALL_PARTS_NAME_ = {"nose", "neck", "r_sho", "r_elb", "r_wri", "l_sho", 
                                                    "l_elb", "l_wri", "r_hip", "r_knee", "r_ank", "l_hip", 
                                                    "l_knee", "l_ank", "r_eye", "l_eye", "r_ear", "l_ear"};

    };

}


#endif //HandsUpVOTINGDETECT_OPENPOSELITE_H
