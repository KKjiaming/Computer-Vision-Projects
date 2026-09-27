//
// Created by kun on 2022/4/30.
//

#ifndef HandsUpVOTING_DETECT_MODULE_H
#define HandsUpVOTING_DETECT_MODULE_H

#include "Common.h"
#include <vector>
#include "opencv2/opencv.hpp"

namespace smartmore {

    //!
    //! \struct HandsUpResult
    //! \param id track id of person
    //! \param keyPoints store human body keypoints
    //!    18 body keypoints coordinates {'nose', 'neck', 'right shoulder', 'right elbow', 'right wristle', 'left shoulder', 'left elbow', 'left wristle', 'right hip',
    //!                                   'right knee', 'right ankle', 'left hip', 'left knee', 'left ankle', 'right eye', 'left eye', 'right ear', 'left ear'}
    //!    example: {{1, 2}, {3, 5}, {10, 10}, {-1, -1}, {-1, -1}, {11, 33}, {7, 7}, {3, 15}, ...}
    //!    {-1, -1} means no keypoint not detected (right elbow and right wristle not detected)
    //! \param bbox box of person
    //! \param lefthandUp whether left hand is up (true or false)
    //! \param righhandUp whether right hand is up (true or false)
    //!
    struct HandsUpResult {
        int id;
        std::vector<cv::Point> keyPoints;
        cv::Rect bbox;
        bool lefthandUp;
        bool righthandUp;
    };

    //!
    //! \struct HandsUpState
    //! \param id track id of person
    //! \param lefthandUp whether left hand is up (true or false)
    //! \param righhandUp whether right hand is up (true or false)
    //!
    struct HandsUpState {
        int id;
        bool lefthandUp;
        bool righthandUp;
    };

    //!
    //! \struct Request
    //! \param image model input image
    //!
    struct HandsUpDet_Request {
        cv::Mat image;
    };

    //!
    //! \struct HandsUpDet_Response
    //! \handsUpResults store handsup information of people.
    //! \handsUpNums number of person whose hand is up
    //!
    struct HandsUpDet_Response {
        std::vector<HandsUpResult> handsUpResults;
        int handsUpNums;
    };

    class HandsUpDetectModule {
    public:
        HandsUpDetectModule();

        ~HandsUpDetectModule();

        //!
        //! \brief Init HandsUpDetectModule with given model path.
        //!
        //! \param model_path The path of input ObjectDetector model.
        //! \param use_gpu Whether use gpu in subsequent inference.
        //! \param device_id ID of gpu used.(-1 for auto choose, default device 0)
        //!
        //! \return The status of Initialization. OK for success, else non-zero.
        //!
        //! In particular, this function is called prior to any call to Run().
        //!
        ResultCode Init(const std::string &model_path, bool use_gpu, int device_id);

        //!
        //! \brief Do inference.
        //!
        //! \param req The origin request.
        //! \param rsp The response generated from Run.
        //!
        //! \return The status of Inference. OK for success, else non-zero.
        //!
        ResultCode run(const HandsUpDet_Request &req, HandsUpDet_Response &rsp);

        //!
        //! \brief get hands up numbers base on analysis records during voting
        //! \param records a array all hands up states in a continuous frames during voting
        //!
        int get_handsUp_Nums(const std::vector<std::vector<HandsUpState>> &records);

    private:
        class Impl;

        std::shared_ptr<Impl> _impl;
    };

}

#endif //HandsUpVOTING_DETECT_MODULE_H
