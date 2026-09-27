//
// Created by kun on 2022/4/9.
//

#ifndef HANDSUP_DETECT_OrtInferEngine_H
#define HANDSUP_DETECT_OrtInferEngine_H

#include "../common/data_types.h"
#include "../../include/Common.h"
#include "onnxruntime_cxx_api.h"
//#include "cuda_provider_factory.h"
#include <iostream>

namespace smartmore {

    class OrtInferEngine {
    public:
        OrtInferEngine();

        virtual ~OrtInferEngine();

    protected:

        virtual ResultCode Init(const std::vector<unsigned char>& model_data, bool use_gpu, int device_id);

        virtual void Inference(std::vector<Tensor>& in_data, std::vector<TensorNode>& out_data);

    protected:
        std::shared_ptr<Ort::Session> pSession;
        std::shared_ptr<Ort::Env> env;
        Ort::SessionOptions sessionOptions;
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<char*> input_names;
        std::vector<char*> output_names;
    };

}


#endif //HANDSUP_DETECT_OrtInferEngine_H
