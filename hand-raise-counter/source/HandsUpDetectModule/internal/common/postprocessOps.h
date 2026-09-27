//
// Created by kun on 2022/4/8.
//

#ifndef VIDEOANALYSIS_POSTPROCESSOPS_H
#define VIDEOANALYSIS_POSTPROCESSOPS_H

#include "../../include/Common.h"
#include "data_types.h"

namespace smartmore {

    float bbox_iou(const BBox &bbox1, const BBox& bbox2);

    void bbox_nms(std::vector<DetObject> &detObjects, std::vector<DetObject> &results, float nms_thresh);

    void multiclass_bbox_nms(std::vector<std::vector<DetObject>> &detObjects_vec,
                             std::vector<DetObject> &results, float nms_thresh);

}

#endif //VIDEOANALYSIS_POSTPROCESSOPS_H
