//
// Created by kun on 2022/4/13.
//

#include "utility.h"

namespace smartmore {

    void visualizeRoiArea(cv::Mat &visualImage, const cv::Rect& roi, const cv::Scalar& color) {
        if (roi.width < 60 || roi.height < 60) {

        }
        int gap = 10;
        int xN = roi.width / gap;
        for (int i = 0; i < xN; i += 2) {
            int x1 = roi.x + i * gap;
            int x2 = x1 + gap;
            cv::line(visualImage, cv::Point(x1, roi.y),
                     cv::Point(x2, roi.y), color, 5);
            cv::line(visualImage, cv::Point(x1, roi.y+roi.height),
                     cv::Point(x2, roi.y+roi.height), color, 5);
        }
        int yN = roi.height / gap;
        for (int i = 0; i < yN; i += 2) {
            int y1 = roi.y + i * gap;
            int y2 = y1 + gap;
            cv::line(visualImage, cv::Point(roi.x, y1),
                     cv::Point(roi.x, y2), color, 5);
            cv::line(visualImage, cv::Point(roi.x+roi.width, y1),
                     cv::Point(roi.x+roi.width, y2), color, 5);
        }

        int baseline;
        float ratio = 0.75;
        const std::string drawText = "detect ROI";
        cv::Size textSize = cv::getTextSize(
                drawText, cv::FONT_HERSHEY_SIMPLEX, 1.2f, 2, &baseline);
        auto textBox = cv::Rect(roi.x, roi.y - textSize.height,
                                textSize.width * ratio, textSize.height);
        cv::rectangle(visualImage, textBox, color, -1);

        cv::putText(visualImage, drawText,
                    cv::Point2f(textBox.x + textSize.width * 0.03, textBox.y + textSize.height * ratio),
                    cv::FONT_HERSHEY_SIMPLEX, /*text size*/ 0.7,
                    cv::Scalar(255, 255, 255), 2);

    }

    void visualizeAllPoses(cv::Mat &image, const std::vector<HandsUpResult> &handsUpResults) {
        static const std::vector<std::pair<int, int>> BODY_PARTS_KPT_IDS_ = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
                                                                      {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};
        static const std::vector<std::pair<int, int>> BODY_PARTS_PAF_IDS_ = {{12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1}, {2, 3}, {4, 5},
                                                                      {6, 7}, {8, 9}, {10, 11}, {28, 29}, {30, 31}, {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27}};

        for (const auto &result : handsUpResults) {
            auto &keypoints = result.keyPoints;
            for (size_t part_id = 0; part_id + 2 < BODY_PARTS_PAF_IDS_.size(); ++part_id) {
                int kpt_a_id = BODY_PARTS_KPT_IDS_[part_id].first;
                int global_kpt_a_id = keypoints[kpt_a_id].x;
                if (global_kpt_a_id != -1) 
                    cv::circle(image, keypoints[kpt_a_id], 3, cv::Scalar(0, 255, 255), -1);
                int kpt_b_id = BODY_PARTS_KPT_IDS_[part_id].second;
                int global_kpt_b_id = keypoints[kpt_b_id].x;
                if (global_kpt_b_id != -1)
                    cv::circle(image, keypoints[kpt_b_id], 3, cv::Scalar(0, 255, 255), -1);
                if (global_kpt_a_id != -1 && global_kpt_b_id != -1)
                    cv::line(image, keypoints[kpt_a_id], keypoints[kpt_b_id], cv::Scalar(0, 255, 255), 2);
            }
            cv::rectangle(image, result.bbox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            char info[512];
            sprintf(info, "id: %d", result.id);
            cv::putText(image, info,
                        cv::Point(result.bbox.x, result.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
    }

    void visualizeHandsUpResults(cv::Mat &image, const std::vector<HandsUpResult> &handsUpResults, int handsUpNums, long long frame_idx) {
        static const std::vector<std::pair<int, int>> LEFT_KPT_IDS_ = {{1, 5}, {5, 6}, {6, 7}};
        static const std::vector<std::pair<int, int>> RIGHT_KPT_IDS_ = {{1, 2}, {2, 3}, {3, 4}};

        for (const auto &result : handsUpResults) {
            auto &keypoints = result.keyPoints;
            if (result.lefthandUp) {
                for (size_t part_id = 2; part_id < LEFT_KPT_IDS_.size(); ++part_id) {
                    int kpt_a_id = LEFT_KPT_IDS_[part_id].first;
                    int global_kpt_a_id = keypoints[kpt_a_id].x;
                    if (global_kpt_a_id != -1) 
                        cv::circle(image, keypoints[kpt_a_id], 3, cv::Scalar(0, 255, 255), -1);
                    int kpt_b_id = LEFT_KPT_IDS_[part_id].second;
                    int global_kpt_b_id = keypoints[kpt_b_id].x;
                    if (global_kpt_b_id != -1)
                        cv::circle(image, keypoints[kpt_b_id], 3, cv::Scalar(0, 255, 255), -1);
                    if (global_kpt_a_id != -1 && global_kpt_b_id != -1)
                        cv::line(image, keypoints[kpt_a_id], keypoints[kpt_b_id], cv::Scalar(0, 255, 255), 2);
                }
            }
            if (result.righthandUp) {
                for (size_t part_id = 2; part_id < RIGHT_KPT_IDS_.size(); ++part_id) {
                    int kpt_a_id = RIGHT_KPT_IDS_[part_id].first;
                    int global_kpt_a_id = keypoints[kpt_a_id].x;
                    if (global_kpt_a_id != -1) 
                        cv::circle(image, keypoints[kpt_a_id], 3, cv::Scalar(0, 255, 255), -1);
                    int kpt_b_id = RIGHT_KPT_IDS_[part_id].second;
                    int global_kpt_b_id = keypoints[kpt_b_id].x;
                    if (global_kpt_b_id != -1)
                        cv::circle(image, keypoints[kpt_b_id], 3, cv::Scalar(0, 255, 255), -1);
                    if (global_kpt_a_id != -1 && global_kpt_b_id != -1)
                        cv::line(image, keypoints[kpt_a_id], keypoints[kpt_b_id], cv::Scalar(0, 255, 255), 2);
                }
                
            }
            std::string info = "id: " + std::to_string(result.id);
            if (result.lefthandUp && result.righthandUp)
                info += " double";
            else if (result.lefthandUp)
                info += " left";
            else if (result.righthandUp)
                info += " right";
            
            auto display_color = (result.lefthandUp || result.righthandUp) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);

            cv::putText(image, info,
                        cv::Point(result.bbox.x, result.bbox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        display_color, 2);

            display_color = (result.lefthandUp || result.righthandUp) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            
            cv::rectangle(image, result.bbox, display_color, 2, cv::LINE_AA);
        }
        char info[512];
        sprintf(info, "handsUp Number : %d   frame : %lld", handsUpNums, frame_idx);
        cv::putText(image, info,
                    cv::Point(5, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 255), 2);
    }

}
