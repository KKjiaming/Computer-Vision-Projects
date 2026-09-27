#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <HandsUpDetectModule.h>
#include "../internal/OpenPoseLite/Pose.h"

namespace smartmore {

struct PersonByDet 
{
    static int last_id;
    int id;
    BBox bbox;

    explicit PersonByDet(const BBox &box_conf);

    inline void half_xyxy(BBox &box);

    inline void update_id(int _id = -1);
};

class PostProcess
{
public:
    PostProcess();
    ~PostProcess();

    static cv::Rect roundBBox(const BBox &bbox);

    static void group_bbox_poses(const std::vector<PersonByDet> &current_persons, std::vector<Pose> &current_poses);

    static std::pair<bool, float> cal_angle(const cv::Point &elbow, const cv::Point &wrist, const cv::Point &nose);

    static int countHandsUpbyPoses(std::vector<Pose> &current_poses);

    void track_persons(const std::vector<PersonByDet> &previous_persons, 
                       std::vector<PersonByDet> &current_persons, 
                       float threshold = 0.2);

private:
    static inline cv::Point2f get_bboxCenter(const BBox &bbox);

private:
    std::unordered_map<int, cv::Point2f> id_boxCenters_;
    std::unordered_map<int, int> id_unMatched_ages_;
};

}

#endif