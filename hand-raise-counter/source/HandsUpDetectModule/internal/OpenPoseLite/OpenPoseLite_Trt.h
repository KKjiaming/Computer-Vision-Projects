//
// Created by kun on 2022/6/20.
//

#ifndef HandsUpVOTINGDETECT_OPENPOSELITE_TRT_H
#define HandsUpVOTINGDETECT_OPENPOSELITE_TRT_H

#include "Common.h"
#include "opencv2/opencv.hpp"
#include "OpenPoseLite.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "../InferEngine/Utils.h"

namespace smartmore {
    class OpenPoseLite_Trt : public OpenPoseLite {
    public:
        OpenPoseLite_Trt();
        ~OpenPoseLite_Trt();
        ResultCode Init(const std::vector<unsigned char> &model_data,
                        const std::string &model_name,
                        bool use_gpu, bool useFP16, int device_id);

        ResultCode run(const cv::Mat &image, std::vector<Pose> &current_poses);

    private:

        void postprocess(const float *heatMapsData,
                         const float *pafsData,
                         std::vector<Pose> &current_poses);

        static bool checkTensorShape(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);

    private:

        void doInference();

        void initEngine(const std::vector<unsigned char> &modelData);

        void allocateMemory();

        static inline std::vector<char> loadTimingCacheFile(const std::string &inFileName);

        static inline void saveTimingCacheFile(const std::string &outFileName, const nvinfer1::IHostMemory *blob);

    private:
        bool useFP16_;
        int maxBatchSize_;
        int device_id_;
        std::string model_name_;
        std::string cache_file_;

        const std::string inputName_ = "data";
        const std::string output1Name_ = "stage_1_output_1_heatmaps";
        const std::string output2Name_ = "stage_1_output_0_pafs";

        Logger logger;

        // the object of TensorRT.
        std::unique_ptr <nvinfer1::IRuntime> inferRuntime_ = nullptr;
        std::unique_ptr <nvinfer1::ICudaEngine> inferEngine_ = nullptr;
        std::unique_ptr <nvinfer1::IExecutionContext> inferContext_ = nullptr;

        // cuda stream.
        cudaStream_t stream_{};

        int32_t opt_height_ = 512;
        int32_t opt_width_ = 912;
        int32_t opt_minHeight_ = 256;
        int32_t opt_minWidth_ = 256;
        int32_t opt_maxHeight_ = 1024;
        int32_t opt_maxWidth_ = 1024;

        int64_t curInputSize_;
        int64_t curOutputSize1_;
        int64_t curOutputSize2_;

        nvinfer1::Dims outputDims1_;
        nvinfer1::Dims outputDims2_;

        // device inputs and outputs memory
        float *inferInputBuffer_ = nullptr;
        float *inferOutputBuffer1_ = nullptr;
        float *inferOutputBuffer2_ = nullptr;
        float *inferBuffers_[3];

        // network inputs and outputs
        float *trt_inferInput_ = nullptr;
        float *trt_inferOutput1_ = nullptr;
        float *trt_inferOutput2_ = nullptr;

    };

}


#endif //HandsUpVOTINGDETECT_OPENPOSELITE_TRT_H
