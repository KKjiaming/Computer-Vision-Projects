//
// Created by kun on 2022/6/20.
//

#include "OpenPoseLite.h"
#include "../common/preprocessOps.h"
#include <unordered_set>
#include <cmath>
#include <fstream>
#include <omp.h>

using namespace smartmore;

OpenPoseLite::OpenPoseLite() = default;

OpenPoseLite::~OpenPoseLite() = default;

void OpenPoseLite::post_process(const float* heatMapsData,
                                const float* pafsData,
                                int heatMapOffset, int nHeatMaps,
                                int pafOffset, int nPafs,
                                int featureMapHeight, int featureMapWidth,
                                float conf_thresh, float keypoint_thresh,
                                std::vector<Pose> &current_poses) {

    // auto fout1 = fopen("cppheatmap.txt", "w");
    // for (size_t i = 0; i < nHeatMaps * featureMapHeight * featureMapWidth; ++i)
    //     fprintf(fout1, "%.10f ", heatMapsData[i]);
    // fprintf(fout1, "\n");
    // fclose(fout1);
    // auto fout2 = fopen("cpppafs.txt", "w");
    // for (size_t i = 0; i < nPafs * featureMapHeight * featureMapWidth; ++i)
    //     fprintf(fout2, "%.10f ", pafsData[i]);
    // fprintf(fout2, "\n");
    // fclose(fout2);

    std::vector<cv::Mat> heatMaps(nHeatMaps);

    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                              reinterpret_cast<void*>(
                                      const_cast<float*>(
                                              heatMapsData + i * heatMapOffset)));
    }
    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(nPafs);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                          reinterpret_cast<void*>(
                                  const_cast<float*>(
                                          pafsData + i * pafOffset)));
    }
    resizeFeatureMaps(pafs);

    std::vector<std::vector<Peak>> all_keyPoints_by_type(num_keyPoints_);
    {
	    // omp_set_num_threads(2);
// #pragma omp parallel for
        for (int heatMapIdx = 0; heatMapIdx < num_keyPoints_; ++heatMapIdx) {
            extractKeyPoints(heatMaps, all_keyPoints_by_type, heatMapIdx);
        }
    }

    int keyPoint_num = 0;
    for (int heatMapIdx = 1; heatMapIdx < num_keyPoints_; heatMapIdx++) {
        keyPoint_num += static_cast<int>(all_keyPoints_by_type[heatMapIdx - 1].size());
        for (auto& peak : all_keyPoints_by_type[heatMapIdx]) {
            peak.idx += keyPoint_num;
        }
    }

    // for (const auto &points : all_keyPoints_by_type) {
    //     printf("\033[31m===================\n\033[0m");
    //     for (const auto &pt : points) {
    //         printf("\033[32m%f %f %f %d\n\033[0m", pt.x, pt.y, pt.score, pt.idx);
    //     }
    // }

    std::vector<Peak> all_keyPoints;
    std::vector<std::vector<float>> pose_entries;
    group_keyPoints(all_keyPoints_by_type, pafs, all_keyPoints, pose_entries);

    // printf("\033[31m===================\n\033[0m");
    // for (const auto &pose_entry : pose_entries) {
    //     for (auto &v : pose_entry) {
    //         printf("\033[32m%f \033[0m", v);
    //     }
    //     printf("\n");
    // }

    for (size_t kpt_id = 0; kpt_id < all_keyPoints.size(); ++kpt_id) {
        all_keyPoints[kpt_id].x = (all_keyPoints[kpt_id].x * stride_ / upsample_ratio_ - pads_[1]) / scale_;
        all_keyPoints[kpt_id].y = (all_keyPoints[kpt_id].y * stride_ / upsample_ratio_ - pads_[0]) / scale_;
    }

    for (auto &pose_entry : pose_entries) {
        if (pose_entry.empty())
            continue;
        if (pose_entry[18] < conf_thresh) // 卡置信度
            continue;
        std::vector<cv::Point> pose_keypoints(num_keyPoints_, cv::Point2f(-1, -1));
        // for (int kpt_id = 0; kpt_id < num_keyPoints_; ++kpt_id) {
        for (int kpt_id : UPPER_PARTS_IDS) {
            if (pose_entry[kpt_id] != -1) {
                int k = cvRound(pose_entry[kpt_id]);
                pose_keypoints[kpt_id] = cv::Point(all_keyPoints[k].x, all_keyPoints[k].y);
            }
        }
        // printf("\033[31m%f\n\033[0m", pose_entry[num_keyPoints_]);
        // for (const auto &keypoint : pose_keypoints) {
        //     std::cout << "\033[32m" << keypoint << "\n\033[0m";
        // }
        // filter pose by keypoint confidence
        if (keypointfilter(pose_entry, all_keyPoints, keypoint_thresh))
            current_poses.emplace_back(Pose(pose_keypoints, pose_entry[num_keyPoints_]));
    }
    // std::cout << "\033[33m=============\033[0m" << std::endl;
    // std::cout << "\n";
}

bool OpenPoseLite::keypointfilter(std::vector<float> &pose_entry, const std::vector<Peak> &all_keypoints, float keypoint_threshold) {
    // all_parts = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho', 
    //             'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip', 
    //             'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']
    /***
    1. Limite confidence of wrist and elbow of left and right, confidence score is 0.4
    2. If three or more points in 1. are filtered, this pose_entries is regarded as a fake pose and return false
    ***/
    int filter_num = 0;
    std::vector<int> partsIdx{3, 4, 6, 7};
    for (int kpt_id : partsIdx) {
        auto idx = cvRound(pose_entry[kpt_id]);
        if (idx == -1 || all_keypoints[idx].score < keypoint_threshold) {
            pose_entry[kpt_id] = -1;
            ++ filter_num;
        }
    }
    return filter_num < 3;
}

void OpenPoseLite::extractKeyPoints(const std::vector<cv::Mat>& heatMaps,
                                    std::vector<std::vector<Peak>>& all_keyPoints_by_type,
                                    int heatMapIdx, float minPeakDistance, float threshold) {
    std::vector<cv::Point> peakPoints;
    const cv::Mat& heatMap = heatMaps[heatMapIdx];
    int h = (int)heatMap.rows, w = (int)heatMap.cols;

    float val = heatMap.at<float>(0, 0);
    float right_val = heatMap.at<float>(0, 1);
    float down_val = heatMap.at<float>(1, 0);
    threshold += 5e-6f;  // pth 和 onnx 模型的精度差异 0.9999 和 1.00003 主要也是置信度比较低的
    if (val > threshold && val > right_val && val > down_val)
        peakPoints.emplace_back(0, 0);

    val = heatMap.at<float>(0, w-1);
    float left_val = heatMap.at<float>(0, w-2);
    down_val = heatMap.at<float>(1, w-1);
    if (val > threshold && val > left_val && val > down_val)
        peakPoints.emplace_back(w-1, 0);

    val = heatMap.at<float>(h-1, 0);
    float up_val = heatMap.at<float>(h-2, 0);
    right_val = heatMap.at<float>(h-1, 1);
    if (val > threshold && val > up_val && val > right_val)
        peakPoints.emplace_back(0, h-1);

    val = heatMap.at<float>(h-1, w-1);
    up_val = heatMap.at<float>(h-2, w-1);
    left_val = heatMap.at<float>(h-1, w-2);
    if (val > threshold && val > up_val && val > left_val)
        peakPoints.emplace_back(w-1, h-1);

    for (int i = 1; i < h-1; ++i) {
        val = heatMap.at<float>(i, 0);
        if (val < threshold)
            continue;
        up_val = heatMap.at<float>(i-1, 0);
        right_val = heatMap.at<float>(i, 1);
        down_val = heatMap.at<float>(i+1, 0);
        if (val > up_val && val > right_val && val > down_val) {
            peakPoints.emplace_back(0, i);
            ++i;
        }
    }
    for (int i = 1; i < h-1; ++i) {
        val = heatMap.at<float>(i, w-1);
        if (val < threshold)
            continue;
        up_val = heatMap.at<float>(i-1, w-1);
        left_val = heatMap.at<float>(i, w-2);
        down_val = heatMap.at<float>(i+1, w-1);
        if (val > threshold && val > up_val && val > left_val && val > down_val) {
            peakPoints.emplace_back(w-1, i);
            ++i;
        }
    }

    for (int i = 1; i < w-1; ++i) {
        val = heatMap.at<float>(0, i);
        if (val < threshold)
            continue;
        left_val = heatMap.at<float>(0, i - 1);
        right_val = heatMap.at<float>(0, i + 1);
        down_val = heatMap.at<float>(1, i);
        if (val > threshold && val > left_val && val > right_val && val > down_val) {
            peakPoints.emplace_back(i, 0);
            ++i;
        }
    }
    for (int i = 1; i < w-1; ++i) {
        val = heatMap.at<float>(h-1, i);
        if (val < threshold)
            continue;
        left_val = heatMap.at<float>(h-1, i-1);
        right_val = heatMap.at<float>(h-1, i+1);
        up_val = heatMap.at<float>(h-2, i);
        if (val > threshold && val > left_val && val > right_val && val > up_val) {
            peakPoints.emplace_back(i, h-1);
            ++i;
        }
    }

    for (int i = 1; i < h-1; ++i) {
        for (int j = 1; j < w-1; ++j) {
            val = heatMap.at<float>(i, j);
            if (val < threshold)
                continue;
            
            left_val = heatMap.at<float>(i, j-1);
            right_val = heatMap.at<float>(i, j+1);
            up_val = heatMap.at<float>(i-1, j);
            down_val = heatMap.at<float>(i+1, j);
            if (val > left_val && val > right_val && val > up_val && val > down_val) {
                peakPoints.emplace_back(j, i);
                ++j;  // 确定当前点是峰值，那么下一个点一定不是峰值
            }
        }
    }

    std::sort(peakPoints.begin(), peakPoints.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    std::vector<bool> isActualPeak(peakPoints.size(), true);
    int peakCounter = 0;
    auto& peaksWithScoreAndID = all_keyPoints_by_type[heatMapIdx];
    for (size_t i = 0; i < peakPoints.size(); i++) {
        if (isActualPeak[i]) {
            for (size_t j = i + 1; j < peakPoints.size(); j++) {
                if (sqrt((peakPoints[i].x - peakPoints[j].x) * (peakPoints[i].x - peakPoints[j].x) +
                         (peakPoints[i].y - peakPoints[j].y) * (peakPoints[i].y - peakPoints[j].y)) < minPeakDistance) {
                    isActualPeak[j] = false;
                }
            }
            peaksWithScoreAndID.emplace_back(peakCounter++, peakPoints[i].x, peakPoints[i].y, heatMap.at<float>(peakPoints[i]));
        }
    }

}

void OpenPoseLite::group_keyPoints(const std::vector<std::vector<Peak>> &all_keyPoint_by_type,
                                   const std::vector<cv::Mat> &pafMaps,
                                   std::vector<Peak> &all_keyPoints,
                                   std::vector<std::vector<float>> &pose_entries,
                                   float pose_entry_size,
                                   float min_paf_score) {

    for (const auto &keyPoints : all_keyPoint_by_type) {
        for (const auto &keyPoint : keyPoints) {
            all_keyPoints.emplace_back(keyPoint);
        }
    }
    // for (const auto &p : all_keyPoints) {
    //     std::cout << p << std::endl;
    // }
    int points_per_limb = 10;
    for (size_t part_id = 0; part_id < BODY_PARTS_KPT_IDS_.size(); ++part_id) {
        std::vector<cv::Mat> part_pafs{pafMaps[BODY_PARTS_PAF_IDS_[part_id].first],
                                       pafMaps[BODY_PARTS_PAF_IDS_[part_id].second]};
        auto kpts_a = all_keyPoint_by_type[BODY_PARTS_KPT_IDS_[part_id].first];
        auto kpts_b = all_keyPoint_by_type[BODY_PARTS_KPT_IDS_[part_id].second];
        int n = (int) kpts_a.size(), m = (int) kpts_b.size();

        if (n == 0 || m == 0)
            continue;

        std::vector<cv::Point2f> tmp;
        for (int i = 0; i < n; ++i)
            tmp.emplace_back(kpts_a[i].x, kpts_a[i].y);
        std::vector<std::vector<cv::Point2f>> a(m, tmp);
        std::vector<std::vector<cv::Point2f>> b(m);
        for (int i = 0; i < m; ++i) {
            b[i].resize(n, cv::Point2f(kpts_b[i].x, kpts_b[i].y));
        }

        const int mn = m * n;

        std::vector<cv::Point2f> vec_raw(mn);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                vec_raw[i * n + j] = b[i][j] - a[i][j];
            }
        }

        std::vector<std::vector<cv::Point>> points(mn, std::vector<cv::Point>(points_per_limb));
        float step_alpha = 1.f / (static_cast<float>(points_per_limb) - 1);
        for (int i = 0; i < mn; ++i) {
            for (int j = 0; j < points_per_limb; ++j) {
                auto point = (step_alpha * vec_raw[i] * static_cast<float>(j) + a[i / n][i % n]);
                // 对齐这里有取整
                points[i][j].x = cvRound(point.x);
                points[i][j].y = cvRound(point.y);
            }
        }

        // for (int i = 0; i < mn; ++i) {
        //     for (int j = 0; j < points_per_limb; ++j) {
        //         std::cout << points[i][j] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        std::vector<std::vector<cv::Point2f>> field(mn, std::vector<cv::Point2f>(points_per_limb));
        for (int i = 0; i < mn; ++i) {
            for (int j = 0; j < points_per_limb; ++j) {
                float paf_v1 = part_pafs[0].at<float>(points[i][j]);
                float paf_v2 = part_pafs[1].at<float>(points[i][j]);
                field[i][j] = cv::Point2f(paf_v1, paf_v2);
            }
        }

        std::vector<cv::Point2f> vec(mn);

        for (int i = 0; i < mn; ++i) 
            vec[i] = vec_raw[i] / (cv::norm(vec_raw[i]) + 1e-6f);

        std::vector<std::tuple<int, int, float>> affinity_idx_scores;

        for (int i = 0; i < mn; ++i) {
            float valid_num = 0;
            float score_sum = 0;
            for (int j = 0; j < points_per_limb; ++j) {
                float score = field[i][j].dot(vec[i]);
                if (score > min_paf_score) {
                    score_sum += score;
                    ++ valid_num;
                }
            }
            float affinity_score = score_sum / (valid_num + 1e-6f);
            float success_ratio = valid_num / static_cast<float>(points_per_limb);
            if (affinity_score > 0 && success_ratio > 0.8f) {
                // 这里出现了float 比较造成的对齐问题  8 / 10 > 0.8  都设成 float
                affinity_idx_scores.emplace_back(i % n, i / n, affinity_score);
            }
        }

        if (affinity_idx_scores.empty())
            continue;

        connection_nms(affinity_idx_scores);

        std::vector<std::tuple<int, int, float>> connections;
        for (const auto &[a_idx, b_idx, affinity_score] : affinity_idx_scores) 
            connections.emplace_back(kpts_a[a_idx].idx, kpts_b[b_idx].idx, affinity_score);

        if (connections.empty())
            continue;

        if (part_id == 0) {
            for (size_t i = 0; i < connections.size(); ++i) {
                pose_entries.emplace_back(std::vector<float>(pose_entry_size, -1));
                auto [c_i, c_j, score] = connections[i];
                pose_entries[i][BODY_PARTS_KPT_IDS_[0].first] = c_i;
                pose_entries[i][BODY_PARTS_KPT_IDS_[0].second] = c_j;
                pose_entries[i][pose_entry_size-1] = 2;
                pose_entries[i][pose_entry_size-2] = all_keyPoints[c_i].score + all_keyPoints[c_j].score + score;
            }
        } else if (part_id == 17 || part_id == 18) {
            auto kpt_a_id = BODY_PARTS_KPT_IDS_[part_id].first;
            auto kpt_b_id = BODY_PARTS_KPT_IDS_[part_id].second;
            for (size_t i = 0; i < connections.size(); ++i) {
                auto [c_i, c_j, score] = connections[i];
                for (size_t j = 0; j < pose_entries.size(); ++j) {
                    if (pose_entries[j][kpt_a_id] == c_i && pose_entries[j][kpt_b_id] == -1) {
                        pose_entries[j][kpt_b_id] = c_j;
                    } else if (pose_entries[j][kpt_b_id] == c_j && pose_entries[j][kpt_a_id] == -1) {
                        pose_entries[j][kpt_a_id] = c_i;
                    }
                }
            }
        } else {
            auto kpt_a_id = BODY_PARTS_KPT_IDS_[part_id].first;
            auto kpt_b_id = BODY_PARTS_KPT_IDS_[part_id].second;
            for (size_t i = 0; i < connections.size(); ++i) {
                int num = 0;
                for (size_t j = 0; j < pose_entries.size(); ++j) {
                    auto [c_i, c_j, score] = connections[i];
                    if (pose_entries[j][kpt_a_id] == c_i) {
                        pose_entries[j][kpt_b_id] = c_j;
                        ++ num;
                        pose_entries[j][pose_entry_size-1] += 1;
                        pose_entries[j][pose_entry_size-2] += all_keyPoints[c_j].score + score; 
                    }
                }
                if (num == 0) {
                    std::vector<float> pose_entry(pose_entry_size, -1);
                    auto [c_i, c_j, score] = connections[i];
                    pose_entry[kpt_a_id] = c_i;
                    pose_entry[kpt_b_id] = c_j;
                    pose_entry[pose_entry_size-1] = 2;
                    pose_entry[pose_entry_size-2] = all_keyPoints[c_i].score + all_keyPoints[c_j].score + score;
                    pose_entries.emplace_back(pose_entry);
                }
            }
        }
        // std::cout << part_id << std::endl;
        // for (const auto &pose_entry : pose_entries) {
        //     for (auto &v : pose_entry) 
        //         printf("\033[32m%.3f \033[0m", v);
        //     printf("\n");
        // }
        // printf("\033[33m==============\n\033[0m");
    }
    // std::cout << std::endl;
    // for (const auto &pose_entry : pose_entries) {
    //     for (auto &v : pose_entry) 
    //         printf("\033[32m%.3f \033[0m", v);
    //     printf("\n");
    // }
    // printf("\033[31m==============\n[0m");
    std::vector<std::vector<float>> filter_entries;
    for (const auto &pose_entry : pose_entries) {
        if (pose_entry[pose_entry_size-1] < 3 || (pose_entry[pose_entry_size-2] / pose_entry[pose_entry_size-1] < 0.2) ) 
            continue;
        filter_entries.emplace_back(pose_entry);
    }
    pose_entries.swap(filter_entries);
}

void OpenPoseLite::connection_nms(std::vector<std::tuple<int, int, float>> &affinity_idx_scores) {
    std::sort(affinity_idx_scores.begin(), affinity_idx_scores.end(), [](const auto &a, const auto &b) -> bool {
        return std::get<2>(a) > std::get<2>(b);
    });
    std::vector<std::tuple<int, int, float>> _affinity_idx_scores;
    std::unordered_set<int> has_kpt_a, has_kpt_b;
    for (const auto &[a_idx, b_idx, affinity_score] : affinity_idx_scores) {
        if (!has_kpt_a.count(a_idx) && !has_kpt_b.count(b_idx)) {
            has_kpt_a.insert(a_idx);
            has_kpt_b.insert(b_idx);
            _affinity_idx_scores.emplace_back(a_idx, b_idx, affinity_score);
        }
    }
    affinity_idx_scores.swap(_affinity_idx_scores);
}

void OpenPoseLite::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
    {
// #pragma omp parallel for
        for (size_t i = 0; i < featureMaps.size(); ++i) {
            cv::resize(featureMaps[i], featureMaps[i], cv::Size(),
                       upsample_ratio_, upsample_ratio_, cv::INTER_CUBIC);
        }
    }
}

void OpenPoseLite::img_preprocess(const cv::Mat &image, cv::Mat &_img) {
    int height = (int)image.rows;
    scale_ = static_cast<float>(height_size_) / static_cast<float>(height);
    cv::resize(image, _img, cv::Size(), scale_, scale_, cv::INTER_LINEAR);
    // cv::imwrite("cppimg1.jpg", image);
    // cv::imwrite("cppimg2.jpg", _img);
    Normalize(_img, mean_, stdDev_, false);

    std::vector<int> min_dims{height_size_, std::max(height_size_, (int)_img.cols)};
    pad_width(_img, 0, min_dims);
    input_h_ = static_cast<int>(_img.rows);
    input_w_ = static_cast<int>(_img.cols);
}

void OpenPoseLite::img_preprocess(const cv::Mat &image, float *data) {
    cv::Mat _img;
    img_preprocess(image, _img);

    // HWC to CHW
    int H = _img.rows, W = _img.cols, C = (int)_img.channels();
#if 1
    int nr = H;
    int nc = W;
    if (_img.isContinuous()) {
        nr = 1;
        nc = nc * H * C;
    }
    int internal = nc / 3;
    int internalX2 = nc / 3 * 2;
    for (int k = 0; k < nr; k++) {
        auto pImage = _img.ptr<float>(k);
        for (int c = 0, idx = 0; c < nc; c += 3, ++idx) {
            data[idx] = *pImage++;
            data[idx + internal] = *pImage++;
            data[idx + internalX2] = *pImage++;
        }
    }
#else
    for (int i = 0; i < H; ++i) {
        auto ptr = _img.ptr<float>(i);
        for (int j = 0; j < W; ++j) {
            for (int k = 0; k < C; ++k) {
                data[k * W * H + i * W + j] = *ptr;
                ++ptr;
            }
        }
    }
#endif
    // auto fout = fopen("data.txt", "w");
    // for (size_t i = 0; i < C*H*W; ++i)
    //     fprintf(fout, "%.10f ", data[i]);
    // fprintf(fout, " \n");
    // fclose(fout);
}

void OpenPoseLite::pad_width(cv::Mat &img, int pad_value, std::vector<int> &min_dims) {
    int h = (int)img.rows, w = (int)img.cols;
    h = std::min(min_dims[0], h);
    auto ratio = ceil(static_cast<float>(min_dims[0]) / static_cast<float>(stride_));
    min_dims[0] = static_cast<int>(round(ratio)) * stride_;
    min_dims[1] = std::max(min_dims[1], w);
    ratio = ceil(static_cast<float>(min_dims[1]) / static_cast<float>(stride_));
    min_dims[1] = static_cast<int>(round(ratio)) * stride_;
    int top = static_cast<int>(floor(min_dims[0] - h) / 2.f);
    int left = static_cast<int>(floor(min_dims[1] - w) / 2.f);
    int bottom = min_dims[0] - h - top;
    int right = min_dims[1] - w - left;
    cv::copyMakeBorder(img, img,
                       top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar::all(pad_value));
    pads_ = {top, left, bottom, right};
}
