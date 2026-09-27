//
// Created by kun on 2022/4/9.
//

#include "OrtInferEngine.h"

using namespace smartmore;

OrtInferEngine::OrtInferEngine() = default;

OrtInferEngine::~OrtInferEngine() = default;

ResultCode OrtInferEngine::Init(const std::vector<unsigned char>& model_data, bool use_gpu, int device_id) {
    try {
        env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_ERROR);

        // OrtTensorRTProviderOptions trt_options{};
        // trt_options.device_id = device_id;
        // trt_options.trt_max_workspace_size = 2147483648;
        // trt_options.trt_max_partition_iterations = 10;
        // trt_options.trt_min_subgraph_size = 5;
        // trt_options.trt_fp16_enable = 0;
        // trt_options.trt_int8_enable = 0;
        // trt_options.trt_int8_use_native_calibration_table = 0;
        // trt_options.trt_engine_cache_enable = 0;
        // trt_options.trt_engine_cache_path = "cache";
        // trt_options.trt_dump_subgraphs = 1;  
        // sessionOptions.AppendExecutionProvider_TensorRT(trt_options);

        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.DisableMemPattern();
        if (use_gpu) {
            auto ret = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, device_id);
            if (ret != nullptr) {
                throw std::runtime_error("cannot append cuda execution provider");
            }
        }
#if WIN32
        pSession = std::make_shared<Ort::Session>(*env, model_data.data(), model_data.size(), sessionOptions);
#else
        pSession = std::make_shared<Ort::Session>(*env, model_data.data(), model_data.size(), sessionOptions);
#endif
        if (pSession == nullptr)
            return ResultCode::NotInited;

        size_t input_count = pSession->GetInputCount();
        size_t output_count = pSession->GetOutputCount();
        for (size_t i = 0; i < input_count; ++i)
            input_names.emplace_back(pSession->GetInputName(i, allocator));
        for (size_t i = 0; i < output_count; ++i)
            output_names.emplace_back(pSession->GetOutputName(i, allocator));
        return ResultCode::Success;
    } catch (const Ort::Exception& e) {
        std::cerr << e.what() << std::endl;
        return ResultCode::InternalError;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return ResultCode::InternalError;
    } catch (...) {
        std::cerr << "Unknown Error!" << std::endl;
        return ResultCode::InternalError;
    }
}

void OrtInferEngine::Inference(std::vector<Tensor> &_in_data, std::vector<TensorNode> &out_data) {
    std::vector<Ort::Value> ort_inputs;
    for (auto & in_data: _in_data) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto in_shape = in_data.Shape();
        auto data_size = (size_t) in_data.Size();
        auto in_tensor = Ort::Value::CreateTensor<float>(memory_info, in_data.mutable_data(),
                                                         data_size, in_shape.data(), in_shape.size());
        if (!in_tensor.IsTensor()) {
            throw std::runtime_error("failed to create input tensor");
        }
        ort_inputs.emplace_back(std::move(in_tensor));
    }
    auto out_tensors = pSession->Run(Ort::RunOptions{nullptr},
                                     input_names.data(), ort_inputs.data(),
                                     ort_inputs.size(), output_names.data(), output_names.size());
    if (out_tensors.empty()) {
        throw std::runtime_error("empty output tensors");
    }
    for (auto & out_tensor : out_tensors) {
        auto out_tensor_type_info = out_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape;
        for (auto shape : out_tensor_type_info.GetShape())
            out_shape.push_back(shape);
        auto p_data = out_tensor.GetTensorMutableData<float>();
        out_data.emplace_back(TensorNode{p_data, out_shape});
    }
}
