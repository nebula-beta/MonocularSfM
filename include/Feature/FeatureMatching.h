#ifndef __FEATURE_MATCHING_H__
#define __FEATURE_MATCHING_H__

#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "Common/Types.h"
#include "Database/Database.h"


namespace MonocularSfM
{



class FeatureMatcher
{
public:
    /**
     * @param database_path         :   数据库路径
     * @param max_num_matches       :   最大匹配数量
     * @param max_distance          :   特征描述之间的最大距离，　如果超过，那么认为不是正确匹配（要求匹配的时候，特征描述子已经归一化）
     * @param distance_ratio        :   1NN < 2NN * distance_ratio那么认为是正确匹配
     * @param cross_check           :   是否开启交叉验证
     */
    FeatureMatcher(const std::string& database_path,
                   const int& max_num_matches = 10240,
                   const double& max_distance = 0.7,
                   const double& distance_ratio = 0.8,
                   const bool& cross_check = true)
                  : database_path_(database_path),
                    max_num_matches_(max_num_matches),
                    max_distance_(max_distance),
                    distance_ratio_(distance_ratio),
                    cross_check_(cross_check) {}


    /**
     * @brief MatchImagePairs   ：   匹配给定的图像对，将匹配结果存在数据库
     *
     *
     * @param image_pairs       ：   图像对
     */
    void MatchImagePairs(const std::vector<std::pair<image_t, image_t>>& image_pairs);
    virtual void RunMatching() = 0;
protected:
    std::string database_path_;
    int max_num_matches_;
    double max_distance_;
    double distance_ratio_;
    bool cross_check_;
    cv::Ptr<Database> database_;

};

class SequentialFeatureMatcher : public FeatureMatcher
{
public:
    /**
     * @param database_path         :   数据库路径
     * @param overlap               :   每张图和前面的几张图片进行匹配
     * @param max_num_matches       :   最大匹配数量
     * @param max_distance          :   特征描述之间的最大距离，　如果超过，那么认为不是正确匹配（要求匹配的时候，特征描述子已经归一化）
     * @param distance_ratio        :   1NN < 2NN * distance_ratio那么认为是正确匹配
     * @param cross_check           :   是否开启交叉验证
     */
    SequentialFeatureMatcher(const std::string& database_path,
                             const int& overlap = 3,
                             const int& max_num_matches = 10240,
                             const double& max_distance = 0.7,
                             const double& distance_ratio = 0.8,
                             const bool& cross_check = true)
                           : FeatureMatcher(database_path, max_num_matches, max_distance, distance_ratio, cross_check),
                             overlap_(overlap) { }
    void RunMatching();
private:
    int overlap_;
};

class BruteFeatureMatcher : public FeatureMatcher
{
public:
    /**
     * @param database_path         :   数据库路径
     * @param max_pairs_size        :   最多max_pairs_size对图像对同时加载进行内存
     * @param max_num_matches       :   最大匹配数量
     * @param max_distance          :   特征描述之间的最大距离，　如果超过，那么认为不是正确匹配（要求匹配的时候，特征描述子已经归一化）
     * @param distance_ratio        :   1NN < 2NN * distance_ratio那么认为是正确匹配
     * @param cross_check           :   是否开启交叉验证
     */
    BruteFeatureMatcher(const std::string& database_path,
                        const int& max_pairs_size = 100,
                        const bool& is_preemtive = true,
                        const int& preemtive_num_features = 100,
                        const int& preemtive_min_num_matches = 4,
                        const int& max_num_matches = 10240,
                        const double& max_distance = 0.7,
                        const double& distance_ratio = 0.8,
                        const bool& cross_check = true)
                       : FeatureMatcher(database_path, max_num_matches, max_distance, distance_ratio, cross_check),
                         max_pairs_size_(max_pairs_size),
                         is_preemtive_(is_preemtive),
                         preemtive_num_features_(preemtive_num_features),
                         preemtive_min_num_matches_(preemtive_min_num_matches){ }
    void RunMatching();

private:

    /**
     * Wu C. Towards Linear-Time Incremental Structure from Motion[C]//
     * International Conference on 3d Vision. IEEE Computer Society, 2013:127-134.
     *
     * @brief PreemptivelyFilterImagePairs  :   使用抢占式匹配，　过滤掉不可能的图像对
     * @param image_pairs                   :   图像对
     */
    std::vector<std::pair<image_t, image_t>> PreemptivelyFilterImagePairs(std::vector<std::pair<image_t, image_t>> image_pairs);

    cv::Mat GetTopScaleDescriptors(const image_t& image_id);
    bool HasTopScaleDescriptorsCache(const image_t& image_id);

    int max_pairs_size_;
    bool is_preemtive_;
    int preemtive_num_features_;
    int preemtive_min_num_matches_;

    std::unordered_map<image_t, cv::Mat> top_scale_descriptors_cache_;

};





// TODO : 利用Vocab Tree进行匹配，　主要目前不知道如何训练得到通用的Vocab Tree
class VocaburaryTreeFeatureMatcher : public FeatureMatcher
{
public:
    void RunMatching();
};





} // namespace MonocularSfM

#endif // __FEATURE_MATCHING_H__
