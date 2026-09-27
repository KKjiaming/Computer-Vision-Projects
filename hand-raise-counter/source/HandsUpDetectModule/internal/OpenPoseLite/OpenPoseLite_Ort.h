//
// Created by kun on 2022/6/20.
//

#ifndef HandsUpVOTINGDETECT_OPENPOSELITE_ORT_H
#define HandsUpVOTINGDETECT_OPENPOSELITE_ORT_H

#include "Common.h"
#include "opencv2/opencv.hpp"
#include "OpenPoseLite.h"
#include "../InferEngine/OrtInferEngine.h"

namespace smartmore {
    class OpenPoseLite_Ort : public OpenPoseLite, OrtInferEngine {
    public:
        OpenPoseLite_Ort();
        ~OpenPoseLite_Ort();
        ResultCode Init(const std::vector<unsigned char> &model_data,
                        const std::string &model_name,
                        bool use_gpu, int device_id);

        ResultCode run(const cv::Mat &image, std::vector<Pose> &current_poses);

    private:

        void postprocess(const TensorNode& heatMapsTensor,
                         const TensorNode& pafsTensor,
                         std::vector<Pose> &current_poses);

        static bool checkTensorShape(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);

    };

}


#endif //HandsUpVOTINGDETECT_OPENPOSELITE_ORT_H
