//
// Created by kun on 2022/4/10.
//

#ifndef YOLOX_TRT_UTILS_H
#define YOLOX_TRT_UTILS_H

#include <fstream>

#define CHECK_CUDA(e)                                           \
  {                                                             \
    if (e != cudaSuccess) {                                     \
      printf("cuda failure: %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                            \
      exit(0);                                                  \
    }                                                           \
  }

struct NvInferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Severity severity = Severity::kWARNING)
            : mReportableSeverity(severity) {}
    void log(Severity severity, const char* msg) noexcept override {
        using namespace nvinfer1;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cout << "[Internal] " + std::string(msg);
                break;
            case Severity::kERROR:
                std::cout << "[Error]" << msg << std::endl;
                break;
            default:
                break;
        }
    }
    Severity getReportableSeverity() const {
        return mReportableSeverity;
    }
    ~Logger() {}

private:
    Severity mReportableSeverity;
};

#endif //YOLOX_TRT_UTILS_H
