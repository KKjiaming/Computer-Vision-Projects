#include "PostProcess.h"
#include <algorithm>
#include <unordered_set>
#include "../internal/common/postprocessOps.h"

using namespace smartmore;


int PersonByDet::last_id = -1;

PersonByDet::PersonByDet(const BBox &box_conf):id(-1), bbox(box_conf)
{
    half_xyxy(bbox);
}

inline void PersonByDet::half_xyxy(BBox &box)
{
    if ((box.ymax - box.ymin) / (box.xmax - box.xmin) > 1.8)
        box.ymax = (box.ymin + box.ymax) / 2;
}

inline void PersonByDet::update_id(int _id)
{
    id = _id;
    if (id == -1) {
        id = PersonByDet::last_id + 1;
        PersonByDet::last_id += 1;
    }
}

PostProcess::PostProcess()
{
    id_boxCenters_.clear();
}

PostProcess::~PostProcess()
{

}

cv::Rect PostProcess::roundBBox(const BBox &bbox)
{
    auto x1 = cvRound(bbox.xmin);
    auto y1 = cvRound(bbox.ymin);
    auto x2 = cvRound(bbox.xmax);
    auto y2 = cvRound(bbox.ymax);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

void PostProcess::group_bbox_poses(const std::vector<PersonByDet> &current_persons, std::vector<Pose> &current_poses)
{
    if (current_poses.empty())
        return;
    std::vector<Pose> filter_poses;
    for (const auto &current_pose : current_poses) {
        auto &nose = current_pose.keyPoints[0];
        for (const auto &current_person : current_persons) {
            const auto &person_bbox = current_person.bbox;
            if (nose.x > person_bbox.xmin && nose.x < person_bbox.xmax && nose.y > person_bbox.ymin && nose.y < person_bbox.ymax) {
                filter_poses.emplace_back(current_pose);
                filter_poses.rbegin()->bbox = roundBBox(current_person.bbox);
                filter_poses.rbegin()->id = current_person.id;
                break;
            }
        }
    }
    current_poses.swap(filter_poses);
}

std::pair<bool, float> PostProcess::cal_angle(const cv::Point &elbow, const cv::Point &wrist, const cv::Point &nose) {
    auto nose2wrist_delta = nose - wrist;
    auto wrist2elbow_delta = elbow - wrist;
    auto nose2wrist = nose2wrist_delta.x * nose2wrist_delta.x + nose2wrist_delta.y * nose2wrist_delta.y;
    auto wrist2elbow = wrist2elbow_delta.x * wrist2elbow_delta.x + wrist2elbow_delta.y * wrist2elbow_delta.y;
    // printf("nose2wrist: %d\nwrist2elbow: %d\n", nose2wrist, wrist2elbow);
    /***
    1. Limite distance between nose and wrist, mustn't be too small.
    2. height of wrist in pic higher than height of elbow in pic.
    3. Limite distance between wrist and elbow, mustn't be too small.
    4. 
    ***/
    if (nose2wrist < 400 || elbow.y < wrist.y || wrist2elbow < 400)
        return std::make_pair(false, 0);
    float angle = 0;
    if (wrist2elbow_delta.x == 0) 
        angle = wrist2elbow_delta.y == 0 ? 0 : 90;
    else {
        angle = std::atan(std::abs(static_cast<float>(wrist2elbow_delta.y) / static_cast<float>(wrist2elbow_delta.x)));
        angle = angle / M_PI * 180;
    }
    angle += 0.00001;
    return std::make_pair(angle > 50, angle);
}

int PostProcess::countHandsUpbyPoses(std::vector<Pose> &current_poses)
{
    int handsUpNums = 0;
    // when no detect nose, bbox is None.
    // ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 
    // 'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip', 
    // 'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']
    for (auto &pose : current_poses) {
        auto nose = pose.keyPoints[0];
        if (nose.x == -1)
            continue;
        auto r_elb = pose.keyPoints[3];
        auto r_wri = pose.keyPoints[4];
        // printf("conf: %f\n", pose.confidence);
        if (r_elb.x != -1 && r_wri.x != -1) {
            auto [r_identifier, angle] = PostProcess::cal_angle(r_elb, r_wri, nose);
            // printf("right: %f\n", angle);
            if (r_identifier) {
                pose.righthandUp = true;
                handsUpNums ++;
            }
        }
        if (!pose.righthandUp) {
            auto l_elb = pose.keyPoints[6];
            auto l_wri = pose.keyPoints[7];
            if (l_elb.x != -1 && l_wri.x != -1) {
                auto [l_identifier, angle] = PostProcess::cal_angle(l_elb, l_wri, nose);
                // printf("left: %f\n", angle);
                if (l_identifier) {
                    pose.lefthandUp = true;
                    handsUpNums ++;
                }
            }
        }
    }
    return handsUpNums;
}

void PostProcess::track_persons(const std::vector<PersonByDet> &previous_persons, 
                                std::vector<PersonByDet> &current_persons, 
                                float threshold)
{
    static int all_unMatched_times = 0;
    std::sort(current_persons.begin(), current_persons.end(), [](const auto &a, const auto &b) -> bool {
        return a.bbox.score > b.bbox.score;
    });
    bool hasMatch = false;
    std::vector<bool> isMatched(previous_persons.size(), false);
    std::unordered_set<int> allocated_previous_persons_ids;
    for (auto &current_person : current_persons) {
        int best_match_id = -1;
        int matched_idx = -1;
        float best_matched_iou = 0;
        for (size_t i = 0; i < previous_persons.size(); i++) {
            if (allocated_previous_persons_ids.count(previous_persons[i].id))
                continue;
            float iou = bbox_iou(current_person.bbox, previous_persons[i].bbox);
            if (iou > best_matched_iou) {
                best_matched_iou = iou;
                best_match_id = previous_persons[i].id;
                matched_idx = i;
            }
        }
        auto current_center = get_bboxCenter(current_person.bbox);
        if (best_matched_iou >= threshold) {
            if (best_match_id != -1)
                allocated_previous_persons_ids.insert(best_match_id);
            isMatched[matched_idx] = true;
            id_boxCenters_[best_match_id] = current_center;
        } else {
            best_match_id = -1;
            float min_distance = 3640;
            for (const auto &[_id, _center] : id_boxCenters_) {
                if (allocated_previous_persons_ids.count(_id))
                    continue;
                float distance = cv::norm(_center - current_center);
                if (distance < min_distance && distance < 100) {
                    min_distance = distance;
                    best_match_id = _id;
                }
            }
        }
        if (best_match_id != -1)
            hasMatch = true;
        current_person.update_id(best_match_id);
    }
    for (size_t i = 0; i < previous_persons.size(); ++i) {
        int id = previous_persons[i].id;
        if (!isMatched[i]) 
            id_unMatched_ages_[id] += 1;
        else if (id_unMatched_ages_.count(id))
            id_unMatched_ages_.erase(id);
    }
    // if (hasMatch) 
    {
        for (auto it = id_unMatched_ages_.begin(); it != id_unMatched_ages_.end();) {
            auto [_id, unMatch_cnt] = *it;
            if (unMatch_cnt >= 250) {
                id_boxCenters_.erase(_id);
                it = id_unMatched_ages_.erase(it);
            } else {
                ++ it;
            }
        }
    }
}

inline cv::Point2f PostProcess::get_bboxCenter(const BBox &bbox)
{
    return cv::Point2f((bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2);
}

