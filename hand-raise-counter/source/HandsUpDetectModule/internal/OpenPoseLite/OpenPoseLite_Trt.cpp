//
// Created by kun on 2022/6/20.
//

#include "OpenPoseLite_Trt.h"
#include "../common/commonFun.h"

using namespace smartmore;


OpenPoseLite_Trt::OpenPoseLite_Trt() = default;

OpenPoseLite_Trt::~OpenPoseLite_Trt() {
    // free gpu memory.
    CHECK_CUDA(cudaFree(inferInputBuffer_));
    CHECK_CUDA(cudaFree(inferOutputBuffer1_));
    CHECK_CUDA(cudaFree(inferOutputBuffer2_));

    // free cpu memory
    delete []trt_inferInput_;
    delete []trt_inferOutput1_;
    delete []trt_inferOutput2_;

    // free cuda stream
    cudaStreamDestroy(stream_);
}

ResultCode OpenPoseLite_Trt::Init(const std::vector<unsigned char> &model_data,
                                  const std::string &model_name,
                                  bool use_gpu, bool useFP16, int device_id) {
    assert (use_gpu);

    int gpus, driverVer;
    CHECK_CUDA(cudaGetDeviceCount(&gpus));
    CHECK_CUDA(cudaDriverGetVersion(&driverVer));
    if (device_id < 0 || device_id >= gpus) {
        throw std::invalid_argument("invalid device_id!");
    }
    useFP16_ = useFP16;
    device_id_ = device_id;
    model_name_ = model_name;
    maxBatchSize_ = 1;

    // init text detection model.
    initEngine(model_data);

    // allocate memory
    allocateMemory();

    return ResultCode::Success;
}

ResultCode OpenPoseLite_Trt::run(const cv::Mat &image, std::vector<Pose> &current_poses) {

    img_preprocess(image, trt_inferInput_);
    // auto t1 = cv::getTickCount();
    doInference();
    // printf("openpose infer time: %f\n", (double)(cv::getTickCount() - t1) / cv::getTickFrequency());

    postprocess(trt_inferOutput1_, trt_inferOutput2_, current_poses);

    return ResultCode::Success;
}

void OpenPoseLite_Trt::postprocess(const float *heatMapsData,
                                   const float *pafsData,
                                   std::vector<Pose> &current_poses) {

    int featureHeight = static_cast<int>(outputDims1_.d[2]);
    int featureWidth = static_cast<int>(outputDims1_.d[3]);
    int heatMapsOffset = featureHeight * featureWidth;
    int nHeatMaps = static_cast<int>(outputDims1_.d[1]);

    assert(outputDims1_.d[2] == outputDims2_.d[2]);
    assert(outputDims1_.d[3] == outputDims2_.d[3]);
    int pafsOffset = heatMapsOffset;
    int nPafs = static_cast<int>(outputDims2_.d[1]);

    post_process(heatMapsData, pafsData,
                 heatMapsOffset, nHeatMaps, pafsOffset, nPafs,
                 featureHeight, featureWidth, 10, 0.36, current_poses);
}

void OpenPoseLite_Trt::doInference() {

    CHECK_CUDA(cudaSetDevice(device_id_));

    curInputSize_ = 3 * input_h_ * input_w_;
    CHECK_CUDA(cudaMemcpyAsync(inferInputBuffer_, trt_inferInput_,
                               sizeof(float) * curInputSize_,
                               cudaMemcpyHostToDevice, stream_));

    nvinfer1::Dims4 inputDims{1, 3, input_h_, input_w_};
    inferContext_->setBindingDimensions(0, inputDims);

    // call TensorRT
    inferContext_->enqueueV2((void **)inferBuffers_, stream_, nullptr);

    outputDims1_ = inferContext_->getBindingDimensions(1);
    curOutputSize1_ = outputDims1_.d[0] * outputDims1_.d[1] * outputDims1_.d[2] * outputDims1_.d[3];
    CHECK_CUDA(cudaMemcpyAsync(trt_inferOutput1_, inferOutputBuffer1_,
                               sizeof(float) * curOutputSize1_,
                               cudaMemcpyDeviceToHost, stream_));

    outputDims2_ = inferContext_->getBindingDimensions(2);
    curOutputSize2_ = outputDims2_.d[0] * outputDims2_.d[1] * outputDims2_.d[2] * outputDims2_.d[3];
    CHECK_CUDA(cudaMemcpyAsync(trt_inferOutput2_, inferOutputBuffer2_,
                               sizeof(float) * curOutputSize2_,
                               cudaMemcpyDeviceToHost, stream_));

    cudaStreamSynchronize(stream_);
}


void OpenPoseLite_Trt::initEngine(const std::vector<unsigned char> &modelData) {

    CHECK_CUDA(cudaSetDevice(device_id_));
    std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter> builder(
            nvinfer1::createInferBuilder(logger));

    inferRuntime_.reset(nvinfer1::createInferRuntime(logger));

    // builder config
    const auto explicitBatch =
            1U << static_cast<uint32_t>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter> network(
            builder->createNetworkV2(explicitBatch));
    std::unique_ptr<nvinfer1::IBuilderConfig, NvInferDeleter> config(
            builder->createBuilderConfig());
    
    config->setMaxWorkspaceSize(1 << 30);

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName_.c_str(),
                            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, opt_minHeight_, opt_minWidth_});
    profile->setDimensions(inputName_.c_str(),
                            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{maxBatchSize_, 3, opt_height_, opt_width_});
    profile->setDimensions(inputName_.c_str(),
                            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchSize_, 3, opt_maxHeight_, opt_maxWidth_});
    config->addOptimizationProfile(profile);

    std::unique_ptr<nvinfer1::ITimingCache, NvInferDeleter> timingCache{nullptr};

    // timing cache
    // cache_file_ = stripPostfix(model_name_) + ".cache";
    cache_file_ = model_name_ + ".cache";
    std::vector<char> loadedCache = loadTimingCacheFile(cache_file_);
    timingCache.reset(config->createTimingCache(
            static_cast<const void *>(loadedCache.data()), loadedCache.size()));

    config->setTimingCache(*timingCache, false);

    // precision
    if (useFP16_)
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // parse model from ONNX
    std::unique_ptr<nvonnxparser::IParser, NvInferDeleter> parser(
            nvonnxparser::createParser(*network, logger));
    parser->parse(modelData.data(), modelData.size());
    for (int i = 0; i < parser->getNbErrors(); ++i) {
        logger.log(nvinfer1::ILogger::Severity::kERROR,
                   parser->getError(i)->desc());
    }

    // build engine
    std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter> plan{
            builder->buildSerializedNetwork(*network, *config)};

    // save timing cache
    std::unique_ptr<nvinfer1::IHostMemory, NvInferDeleter> timingCacheHostData{
            timingCache->serialize()};
    saveTimingCacheFile(cache_file_, timingCacheHostData.get());

    // create context
    inferEngine_.reset(
            inferRuntime_->deserializeCudaEngine(plan->data(), plan->size()));

    inferContext_.reset(inferEngine_->createExecutionContext());
}


void OpenPoseLite_Trt::allocateMemory() {

    CHECK_CUDA(cudaSetDevice(device_id_));
    CHECK_CUDA(cudaStreamCreate(&stream_));

    int64_t inputStreamSize = maxBatchSize_ * 3 * opt_maxHeight_ * opt_maxWidth_;
    CHECK_CUDA(cudaMalloc((void **)&inferInputBuffer_, sizeof(float) * inputStreamSize));
    inferBuffers_[0] = inferInputBuffer_;
    trt_inferInput_ = new float[inputStreamSize];

    int64_t outputStreamSize1 = maxBatchSize_ * 19 * (opt_maxHeight_ / stride_) * (opt_maxWidth_ / stride_);
    CHECK_CUDA(cudaMalloc((void **)&inferOutputBuffer1_, sizeof(float) * outputStreamSize1));
    inferBuffers_[1] = inferOutputBuffer1_;
    trt_inferOutput1_ = new float[outputStreamSize1];

    int64_t outputStreamSize2 = maxBatchSize_ * 38 * (opt_maxHeight_ / stride_) * (opt_maxWidth_ / stride_);
    CHECK_CUDA(cudaMalloc((void **)&inferOutputBuffer2_, sizeof(float) * outputStreamSize2));
    inferBuffers_[2] = inferOutputBuffer2_;
    trt_inferOutput2_ = new float[outputStreamSize2];

}

std::vector<char> OpenPoseLite_Trt::loadTimingCacheFile(const std::string &inFileName) {
    std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
    if (!iFile) {
        std::cout << "[Info]  "
                  << "Could not read timing cache from: " << inFileName
                  << ".\n\tA new timing cache will be generated and written."
                  << "\n\tThis will take some time." << std::endl;
        return std::vector<char>();
    }
    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    std::vector<char> content(fsize);
    iFile.read(content.data(), fsize);
    iFile.close();
    return content;
}

void OpenPoseLite_Trt::saveTimingCacheFile(const std::string &outFileName, const nvinfer1::IHostMemory *blob) {
    std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);
    if (!oFile) {
        std::cout << "[Info] "
                  << "Could not write timing cache to: " << outFileName
                  << std::endl;
        return;
    }
    oFile.write((char *) blob->data(), blob->size());
    oFile.close();
}