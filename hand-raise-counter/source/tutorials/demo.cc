#include "HandsUpDetectModule.h"
#include "utility.h"

using namespace smartmore;

int main(int argc, char *argv[]) {

    printf("\033[33m[Demo] This is HandsUpDetect Demo\n\033[0m");

    // 1. init model
    const std::string model_path = "../../models/handsUpDetModel.smartmore";
    if (!fs::exists(model_path)) {
        printf("%s does not exit, please model path", model_path.c_str());
        return -1;
    }
    HandsUpDetectModule HandsUpDetectModule;
    HandsUpDetectModule.Init(model_path, true, 0);

    // 2. init VideoCapture
    // const std::string video_path = "../../videos/1_IPC-SJZ_main_20220622112556.dav";
    // const std::string video_path = "../../videos/1_IPC-SJZ_main_20220622181746.dav";
    // const std::string video_path = "../../videos/1_IPC_main_20220630173515.dav";
    const std::string video_path = "../../test_data/1_IPC-SJZ_main_20220622181746.dav";

    if (!fs::exists(video_path)) {
        printf("%s does not exit, please check video path\n", video_path.c_str());
        return -1;
    }
    cv::VideoCapture cap(video_path);
    int imgHeight = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int imgWidth = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);

    // save result video
    const std::string save_name = "result.avi";
    std::shared_ptr<cv::VideoWriter> pVideoWriter = std::make_shared<cv::VideoWriter>(
            save_name, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
            fps, cv::Size(imgWidth, imgHeight));

    bool show_img = false;
    int wait_time = 1;
    long long frame_cnt = 0;
    std::vector<std::vector<HandsUpState>> all_records;
    printf("\033[33m[Demo] start fetch video stream!\n\033[0m");
    auto t0 = std::chrono::high_resolution_clock::now();
    while (cap.isOpened()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            printf("\033[32m[Demo] frame is empty or closed, exit!\n\033[0m");
            break;
        }

        // 3. model inference
        HandsUpDet_Request req{frame};
        HandsUpDet_Response rsp;
        HandsUpDetectModule.run(req, rsp);

        // 4. visualize handsUp detections
        visualizeHandsUpResults(frame, rsp.handsUpResults, rsp.handsUpNums, frame_cnt);

        std::vector<HandsUpState> frame_res;
        for (const auto &res : rsp.handsUpResults)
            frame_res.emplace_back(HandsUpState{res.id, res.lefthandUp, res.righthandUp});
        all_records.emplace_back(frame_res);

        auto t2 = std::chrono::high_resolution_clock::now();
        printf("\033[33m[Demo] \033[34mframeIdx: %lld, \033[35mprocess time: %.1f ms, found %d pose, \033[32mhandsUp nums %d\n\033[0m",
               frame_cnt, std::chrono::duration_cast<sm_ms>(t2-t1).count(), static_cast<int>(rsp.handsUpResults.size()), rsp.handsUpNums);

        pVideoWriter->write(frame);

        frame_cnt ++;

        if (show_img) {
            cv::imshow("frame", frame);
            auto key = cv::waitKey(wait_time);
            if (key == 27)
                break;
            else if (key == 32)
                wait_time = 1 - wait_time;
        }
    }

    if (show_img)
        cv::destroyAllWindows();

    // 测试四个时间段
    // get handsUp numbers base on analysis result of continuous frame
    int start_t = 527, end_t = 940;   // 开始投票和结束投票
    if (end_t < all_records.size()) {
        std::vector<std::vector<HandsUpState>> records(all_records.begin()+start_t, all_records.begin()+end_t);
        int handsUp_Number = HandsUpDetectModule.get_handsUp_Nums(records);
        printf("\033[34m[Demo] get handsUp numbers: %d\n\033[0m", handsUp_Number);
    }
    // get handsUp numbers base on analysis result of continuous frame
    start_t = 1786, end_t = 1992;   // 开始投票和结束投票
    if (end_t < all_records.size()) {
        std::vector<std::vector<HandsUpState>> records(all_records.begin()+start_t, all_records.begin()+end_t);
        int handsUp_Number = HandsUpDetectModule.get_handsUp_Nums(records);
        printf("\033[34m[Demo] get handsUp numbers: %d\n\033[0m", handsUp_Number);
    }
    // get handsUp numbers base on analysis result of continuous frame
    if (end_t < all_records.size()) {
        start_t = 2031, end_t = 2238;   // 开始投票和结束投票
        std::vector<std::vector<HandsUpState>> records(all_records.begin()+start_t, all_records.begin()+end_t);
        int handsUp_Number = HandsUpDetectModule.get_handsUp_Nums(records);
        printf("\033[34m[Demo] get handsUp numbers: %d\n\033[0m", handsUp_Number);
    }
    // get handsUp numbers base on analysis result of continuous frame
    start_t = 3027, end_t = 3350;   // 开始投票和结束投票
    if (end_t < all_records.size()) {
        std::vector<std::vector<HandsUpState>> records(all_records.begin()+start_t, all_records.begin()+end_t);
        int handsUp_Number = HandsUpDetectModule.get_handsUp_Nums(records);
        printf("\033[34m[Demo] get handsUp numbers: %d\n\033[0m", handsUp_Number);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("\033[33m[Demo] total process time: %.1f s\n\033[0m",
           std::chrono::duration_cast<sm_s>(t3-t0).count());

    printf("\033[033m[Demo] This is Demo end!\n\033[0m");

    return 0;
}