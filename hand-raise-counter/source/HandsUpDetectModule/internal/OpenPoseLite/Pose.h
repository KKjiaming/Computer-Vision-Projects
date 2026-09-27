//
// Created by kun on 2022/6/22.
//
#ifndef HandsUpVOTINGDETECT_POSE_H
#define HandsUpVOTINGDETECT_POSE_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace smartmore {

    inline double get_alpha(double rate = 30, double cutoff = 1) {
        auto tau = 1. / (2 * M_PI * cutoff);
        auto te = 1. / rate;
        return 1. / (1.f + tau / te);
    }

    struct LowPassFilter {
        bool first = true;
        double x_previous;
        double operator()(double x, double alpha = 0.5) {
            if (first) {
                first = false;
                x_previous = x;
                return x;
            }
            auto x_filtered = alpha * x + (1 - alpha) * x_previous;
            x_previous = x_filtered;
            return x_filtered;
        }
    };

    struct OneEuroFilter {
        bool first;
        double freq;
        double mincutoff;
        double beta;
        double dcutoff;
        LowPassFilter filter_x;
        LowPassFilter filter_dx;
        double x_previous = -1;
        double dx = -1;
        explicit OneEuroFilter(double freq_ = 15, double mincutoff_ = 1, double beta_ = 0.05, double dcutoff_ = 1):
            first(true), freq(freq_), mincutoff(mincutoff_), beta(beta_), dcutoff(dcutoff_), filter_x({true, -1}), filter_dx({true, -1}) {}
        double operator()(double x) {
            if (first) {
                first = false;
                dx = 0;
            } else 
                dx = (x - x_previous) * freq;
            auto dx_smoothed = filter_dx(dx, get_alpha(freq, dcutoff));
            auto cutoff = mincutoff + beta * std::abs(dx_smoothed);
            auto x_filtered = filter_x(x, get_alpha(freq, cutoff));
            x_previous = x;
            return x_filtered;
        }
    };

    class Pose {
    public:
        explicit Pose(const std::vector<cv::Point> &keyPoints_, float confidence_);
        ~Pose();

        static cv::Rect get_bbox(const std::vector<cv::Point> &keyPoints);

        void update_id(int id = -1);

    public:
        bool lefthandUp;
        bool righthandUp;
        int id;
        std::vector<cv::Point> keyPoints;
        float confidence;
        cv::Rect bbox;
        std::vector<std::vector<OneEuroFilter>> filters;

    public:
        constexpr static int num_kpts = 18;
        constexpr static float sigmas[num_kpts] = {.026, .079, .079, .072, .062, .079, .072, .062, 0.107, .087, .089, 0.107, .087, .089, .025, .025, .035, .035};
        constexpr static float vars[num_kpts] = {
            std::pow(.026 * 2 , 2), std::pow(.079 * 2 , 2), std::pow(.079 * 2 , 2), std::pow(.072 * 2 , 2), std::pow(.062 * 2 , 2), std::pow(.079 * 2 , 2), 
            std::pow(.072 * 2 , 2), std::pow(.062 * 2 , 2), std::pow(.107 * 2 , 2), std::pow(.087 * 2 , 2), std::pow(.089 * 2 , 2), std::pow(.107 * 2 , 2), 
            std::pow(.087 * 2 , 2), std::pow(.089 * 2 , 2), std::pow(.025 * 2 , 2), std::pow(.025 * 2 , 2), std::pow(.035 * 2 , 2), std::pow(.035 * 2 , 2)
        };
        static int last_id;
        
    };

    int get_similarity(const Pose& pose1, const Pose& pose2, float threshold = 0.5);

    void track_poses(std::vector<Pose> &previous_poses, std::vector<Pose> &current_poses, int threshold = 3, bool smooth = false);

}

#endif // HandsUpVOTINGDETECT_POSE_H