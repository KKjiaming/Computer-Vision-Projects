//
// Created by kun on 2022/4/30.
//

#include "smdnnmodel/dnnmodel.h"
#include "../internal/common/commonFun.h"
#include "../internal/Yolov5/Yolov5_Ort.h"
#include "../internal/Yolov5/Yolov5_Trt.h"
#include "../internal/OpenPoseLite/OpenPoseLite_Ort.h"
#include "../internal/OpenPoseLite/OpenPoseLite_Trt.h"
#include "PostProcess.h"
#include <fstream>

using namespace smartmore;

class HandsUpDetectModule::Impl {
public:
    Impl():poseEstimator(std::make_shared<OpenPoseLite_Trt>()),
           personDetector(std::make_shared<Yolov5_Trt>()),
           postProcess(std::make_shared<PostProcess>()) {};

    ~Impl() = default;

    ResultCode Init(const std::string &model_path, bool use_gpu, int device_id) {

        std::cout << version() << std::endl;

        smartmore::DNNModel encryptModels;
        encryptModels.UnpackFromFile(model_path.c_str());
        std::string modelVersion;
        encryptModels.GetVersion(modelVersion);
        auto model_count = encryptModels.GetModelNum();
        if (model_count != 2) {
            fprintf(stderr, "[Error] incorrect model number!\n");
            return ResultCode::BadModel;
        }
        std::string model_version;
        encryptModels.GetVersion(model_version);
        printf("\tmodel version: \033[32m%s\033[0m\n", model_version.c_str());

        std::vector<unsigned char> modelData_Vec;
        encryptModels.GetRawModelWithID(0, modelData_Vec);

        // auto tmpStr2 = getModelData(model_path);
        // std::vector<unsigned char> modelData_Vec(tmpStr2.begin(), tmpStr2.end());
        std::string model_name = "poseEstimator-v" + model_version;
        printf("[Info] Init model %s ...\n", model_name.c_str());
        auto ret = poseEstimator->Init(modelData_Vec, model_name, use_gpu, true, device_id);
        if (ret == ResultCode::Success)
            printf("\033[33m[Info] Init model %s success !\n\033[m", model_name.c_str());

        encryptModels.GetRawModelWithID(1, modelData_Vec);
        model_name = "personDetector-v" + model_version;
        printf("[Info] Init model %s ...\n", model_name.c_str());
        ret = personDetector->Init(modelData_Vec, model_name, cv::Size(640, 640),
                                   static_cast<int>(det_class_names.size()), false,
                                   use_gpu, true, device_id);
        if (ret == ResultCode::Success)
            printf("\033[33m[Info] Init model %s success !\n\033[m", model_name.c_str());

        return ret;
    }

    ResultCode run(const HandsUpDet_Request &req, HandsUpDet_Response &rsp) {

        // auto t1 = cv::getTickCount();
        /********************model infer start*******************/
        std::vector<DetObject> det_results;
        auto ret = personDetector->run(req.image, 0.33, 0.2, det_results);
        if (ret != ResultCode::Success) {
            fprintf(stderr, "model infer error!\n");
            return ret;
        }

        std::vector<Pose> current_poses;
        ret = poseEstimator->run(req.image, current_poses);
        if (ret != ResultCode::Success) {
            fprintf(stderr, "model infer error!\n");
            return ret;
        }
        /********************mode infer end********************************/
        // printf("model infer %f\n", (double)(cv::getTickCount()-t1)/cv::getTickFrequency());

        std::vector<PersonByDet> current_persons;
        for (const auto& res : det_results) {
            if (res.cls_idx == 0) {
                current_persons.emplace_back(res.bbox);
            }
        }
        postProcess->track_persons(previous_persons, current_persons);
        previous_persons = current_persons;
        
        // count handsUp numbers in a frame
        rsp.handsUpNums = PostProcess::countHandsUpbyPoses(current_poses);

        if (track) {
            // // track base on keypoints
            // track_poses(previous_poses, current_poses, 3, true);
            // previous_poses = current_poses;
            PostProcess::group_bbox_poses(current_persons, current_poses);
        }

        for (const auto &pose : current_poses) {
            rsp.handsUpResults.emplace_back(HandsUpResult{pose.id, pose.keyPoints, pose.bbox, pose.lefthandUp, pose.righthandUp});
        }
        return ret;
    }

    int get_handsUp_Nums(const std::vector<std::vector<HandsUpState>> &records) {
        std::unordered_map<int, std::vector<int>> track_map;
        int max_person_num = 0;
        for (const auto &record : records) {
            for (const auto &state : record) {
                if (state.lefthandUp || state.righthandUp)
                    track_map[state.id].emplace_back(1);
                else
                    track_map[state.id].emplace_back(0);
            }
            max_person_num = std::max(max_person_num, (int)record.size());
        }
        int res = 0;
        int window = 15, thresh = 6;
        for (auto &item : track_map) {
            auto &tmp = item.second;
            std::vector<int> preSum(tmp.size() + 1);
            for (int i = 1; i <= (int)tmp.size(); ++i )
                preSum[i] = preSum[i-1] + tmp[i-1];
            for (int i = 0; i + window < (int)tmp.size(); ++i) {
                int num = preSum[i+window] - preSum[i];
                if (num >= thresh) {
                    ++res;
                    break;
                }
            }
        }
        res = std::min(res, max_person_num);
        return res;
    }

private:
    std::string version() {
        std::ostringstream versionInfo;
        versionInfo << "\tHandsUpDetect SDK Version: " << "\033[31m" << sdk_version << "\033[0m"
                    << "\n\tCompiled at: " << __DATE__ << " " << __TIME__;
        return versionInfo.str();
    }

private:
    bool track = true;
    const std::string sdk_version = "1.2.0";

    const std::vector<std::string> det_class_names = {
       "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
       "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
       "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
       "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
       "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
       "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
       "hair drier", "toothbrush"
    };

    std::shared_ptr<OpenPoseLite_Trt> poseEstimator;
    std::shared_ptr<Yolov5_Trt> personDetector;
    std::shared_ptr<PostProcess> postProcess;
    std::vector<PersonByDet> previous_persons;
    // std::vector<Pose> previous_poses;

};

HandsUpDetectModule::HandsUpDetectModule():_impl(std::make_shared<Impl>()) {}

HandsUpDetectModule::~HandsUpDetectModule() = default;

ResultCode HandsUpDetectModule::Init(const std::string &model_path, bool use_gpu, int device_id) {
    return _impl->Init(model_path, use_gpu, device_id);
}

ResultCode HandsUpDetectModule::run(const HandsUpDet_Request &req, HandsUpDet_Response &rsp) {
    return _impl->run(req, rsp);
}

int HandsUpDetectModule::get_handsUp_Nums(const std::vector<std::vector<HandsUpState>> &records) {
    return _impl->get_handsUp_Nums(records);
}
