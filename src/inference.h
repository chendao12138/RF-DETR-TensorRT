#pragma once

#define    RET_OK nullptr
#define    USE_CUDA

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <cmath>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


typedef struct _INIT_PARAMs
{
    std::string modelPath;
    std::vector<int> imgSize = { 640, 640 };
	float detThreshold = 0.6;
    bool cudaEnable = false;
    bool trtEnable = !cudaEnable;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
}INIT_PARAMs;


typedef struct _Det
{
    int classId;
    float confidence;
    cv::Rect box;
} Det;

class DETR
{
public:
    DETR();

    ~DETR();

public:
    char* CreateSession(INIT_PARAMs& iParams);

    char* RunSession(cv::Mat& iImg, std::vector<Det>& oResult);

    char* WarmUpSession();

    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<Det>& oResult);

    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
    std::vector<Det> postprocess(const float* dets_b, const float* logits_b,    // [Q,91] logits
        int64_t Q);

    std::vector<std::string> classes{};

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    bool trtEnable;
	std::string modelPath;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    std::vector<int> imgSize;
	float detThreshold;
    float resizeScales;//letterbox scale
};
