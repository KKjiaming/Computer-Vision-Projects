//
// Created by kun on 2022/6/22.
//

#include "Pose.h"

using namespace smartmore;

int Pose::last_id = -1;

Pose::Pose(const std::vector<cv::Point> &keyPoints_, float confidence_):
    lefthandUp(false), righthandUp(false), id(-1), keyPoints(keyPoints_), confidence(confidence_) {
    bbox = get_bbox(keyPoints_);
    for (int i = 0; i < Pose::num_kpts; ++i) {
        filters.push_back({OneEuroFilter(), OneEuroFilter()});
    }
}

Pose::~Pose() = default;

cv::Rect Pose::get_bbox(const std::vector<cv::Point> &keyPoints) {
    std::vector<cv::Point2f> found_keypoints;
    for (const auto &keyPoint : keyPoints) {
        if (keyPoint.x == -1)
            continue;
        found_keypoints.emplace_back(keyPoint);
    }
    return cv::boundingRect(found_keypoints);
}

void Pose::update_id(int id) {
    this->id = id;
    if (this->id == -1) {
        this->id = Pose::last_id + 1;
        Pose::last_id += 1;
    }
}


int smartmore::get_similarity(const Pose& pose1, const Pose& pose2, float threshold) {
    int num_similar_kpt = 0;
    for (int kpt_id = 0; kpt_id < Pose::num_kpts; ++kpt_id) {
        if (pose1.keyPoints[kpt_id].x != -1 && pose2.keyPoints[kpt_id].x != -1) {
            auto delta = pose1.keyPoints[kpt_id] - pose2.keyPoints[kpt_id];
            auto distance = delta.x * delta.x + delta.y * delta.y;
            auto area = std::max(pose1.bbox.area(), pose2.bbox.area());
            auto similarity = std::exp(-distance / (2 * (area + std::numeric_limits<float>::epsilon()) * Pose::vars[kpt_id]));
            if (similarity > threshold) 
                ++ num_similar_kpt;
        }
    }
    return num_similar_kpt;
}


void smartmore::track_poses(std::vector<Pose> &previous_poses, std::vector<Pose> &current_poses, int threshold, bool smooth) {
    std::sort(current_poses.begin(), current_poses.end(), [](const Pose& a, const Pose& b) -> bool {
        return a.confidence > b.confidence;
    });
    std::vector<bool> mask(previous_poses.size(), true);
    for (auto &current_pose : current_poses) {
        int best_matched_id = -1;
        int best_matched_pose_id = -1;
        int best_matched_iou = 0;
        for (int id = 0; id < (int)previous_poses.size(); ++id) {
            if (!mask[id])
                continue;
            auto &previous_pose = previous_poses[id];
            auto iou = get_similarity(current_pose, previous_pose, 0.5f);
            if (iou > best_matched_iou) {
                best_matched_iou = iou;
                best_matched_pose_id = previous_pose.id;
                best_matched_id = id;
            }
        }
        if (best_matched_iou >= threshold)
            mask[best_matched_id] = false;
        else 
            best_matched_pose_id = -1;
        current_pose.update_id(best_matched_pose_id);

        // for (const auto &p : current_pose.keyPoints)
        //     std::cout << "\033[34m" << p << "\n\033[0m";
        // std::cout << "====" << std::endl;
        if (smooth) {
            for (int kpt_id = 0; kpt_id < Pose::num_kpts; ++kpt_id) {
                if (current_pose.keyPoints[kpt_id].x == -1) 
                    continue;
                
                if (best_matched_pose_id != -1 && previous_poses[best_matched_id].keyPoints[kpt_id].x != -1)
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id];
    
                auto x = current_pose.keyPoints[kpt_id].x;
                auto y = current_pose.keyPoints[kpt_id].y;
                current_pose.keyPoints[kpt_id].x = current_pose.filters[kpt_id][0](x);
                current_pose.keyPoints[kpt_id].y = current_pose.filters[kpt_id][1](y);
                // std::cout << "\033[34m" << kpt_id << " " << current_pose.keyPoints[kpt_id] << "\n\033[0m";
            }
            current_pose.bbox = Pose::get_bbox(current_pose.keyPoints);
        }
    }
}

