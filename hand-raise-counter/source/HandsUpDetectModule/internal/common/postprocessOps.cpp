//
// Created by kun on 2022/4/8.
//

#include "postprocessOps.h"

namespace smartmore {

    float bbox_iou(const BBox &bbox1, const BBox &bbox2) {
        auto width_of_intersection = std::min(bbox1.xmax, bbox2.xmax) - std::max(bbox1.xmin, bbox2.xmin);
        auto height_of_intersection = std::min(bbox1.ymax, bbox2.ymax) - std::max(bbox1.ymin, bbox2.ymin);
        if (width_of_intersection < 0 || height_of_intersection < 0)
            return 0;
        auto area_of_intersection = width_of_intersection * height_of_intersection;
        auto bbox1_area = (bbox1.xmax - bbox1.xmin) * (bbox1.ymax - bbox1.ymin);
        auto bbox2_area = (bbox2.xmax - bbox2.xmin) * (bbox2.ymax - bbox2.ymin);
        auto area_of_union = bbox1_area + bbox2_area - area_of_intersection;
        return area_of_intersection / area_of_union;
    }

    void bbox_nms(std::vector<DetObject> &detObjects, std::vector<DetObject> &results, float nms_thresh) {
        std::sort(detObjects.begin(), detObjects.end(), [](const DetObject &a, const DetObject &b) -> bool {
            return a.bbox.score > b.bbox.score;
        });
        std::vector<bool> keep(detObjects.size(), true);
        for (size_t i = 0; i < detObjects.size(); i++) {
            if (keep[i]) {
                for (size_t j = i + 1; j < detObjects.size(); j++)
                    keep[j] = keep[j] && bbox_iou(detObjects[i].bbox, detObjects[j].bbox) < nms_thresh;
                results.emplace_back(detObjects[i]);
            }
        }
    }

    void multiclass_bbox_nms(std::vector<std::vector<DetObject>> &detObjects_vec,
                             std::vector<DetObject> &results, float nms_thresh) {
        for (auto &detObjects : detObjects_vec) {
            if (detObjects.empty())
                continue;
            std::sort(detObjects.begin(), detObjects.end(), [](const DetObject &a, const DetObject &b) -> bool {
                return a.bbox.score > b.bbox.score;
            });
            std::vector<bool> keep(detObjects.size(), true);
            for (size_t i = 0; i < detObjects.size(); i++) {
                if (keep[i]) {
                    for (size_t j = i + 1; j < detObjects.size(); j++)
                        keep[j] = keep[j] && bbox_iou(detObjects[i].bbox, detObjects[j].bbox) < nms_thresh;
                    results.emplace_back(detObjects[i]);
                }
            }
        }
    }

}