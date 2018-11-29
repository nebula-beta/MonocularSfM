#ifndef __FEATURE_UTILS_H__
#define __FEATURE_UTILS_H__
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

namespace MonocularSfM
{





class FeatureUtils
{
public:

    ////////////////////////////////////////////////////////////////////////////////
    // Feature
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 计算图像的SIFT特征点， 及描述子
     *
     * @param image    : 图像
     * @param pts      : [output] 提取出来的特征点
     * @param desc     : [output] 提取出来的特征描述子
     */
    static
    void ExtractFeature(const cv::Mat& image,
                        std::vector<cv::KeyPoint>& kpts,
                        cv::Mat& desc,
                        int num_features = 8024);

    /**
     * 按照尺度从大到小对特征点进行排序, 提取尺度靠前的num_features个特征点
     *
     * @param kpts              : 特征点
     * @param num_features      : 提取尺度靠前的num_features个特征点
     * @param top_scale_kpts    : [output] 尺度靠前的num_features个特征点
     */
    static
    void ExtractTopScaleKeyPoints(const std::vector<cv::KeyPoint> kpts,
                                  const int& num_features,
                                  std::vector<cv::KeyPoint>& top_scale_kpts);

    /**
     * 按照尺度从大到小对特征点进行排序, 提取尺度靠前的num_features个特征点的特征描述子
     *
     * @param kpts                  : 特征点
     * @param descriptors           : 特征描述子
     * @param num_features          : 提取尺度靠前的num_features个特征点
     * @param top_scale_descriptors : [output] 尺度靠前的num_features个特征点所对应的特征描述子
     */
    static
    void ExtractTopScaleDescriptors(const std::vector<cv::KeyPoint> kpts,
                                    const cv::Mat& descriptors,
                                    const int& num_features,
                                    cv::Mat& top_scale_descriptors);


    /**
     * 去除特征点的畸变
     * @param image         : 图像
     * @param K             : 相机内存
     * @param dist_coef		: 相机畸变参数
     * @param pts           : 特征点
     * @param undistort_pts	: [output]去除畸变之后的特征点.  注意 pts.size() >= undistort.pts.size()
     */
    static
    void UndistortFeature(const cv::Mat& image,
                          const cv::Mat& K,
                          const cv::Mat& dist_coef,
                          const std::vector<cv::Point2f>& pts,
                          std::vector<cv::Point2f>& undistort_pts,
                          std::vector<size_t>& index);





    ////////////////////////////////////////////////////////////////////////////////
    // Matches
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * 计算特征点匹配
     * 1NN < distance_ratio * 2NN 的匹配才会被保留
     * @param desc1             : 第一幅图像上的特征描述子
     * @param desc2             : 第二幅图像上的特征描述子
     * @param matches           : [output] 第一幅图像和第二幅图像的特征匹配
     * @param distance_ratio    : 1NN和2NN之间的距离阈值
     */
    static
    void ComputeMatches(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches, const float distance_ratio = 0.8);

    /**
     * 计算特征点的匹配
     * 1NN < distance_ratio * 2NN 的匹配才会被保留, 并且开启交叉验证
     * 如果第一幅图像的第i个特征点与第二幅图像的第j个特征点发生匹配
     * 那么第二幅图像的第j个特征点与第一幅图像的第i个特征点发生匹配
     * @param desc1             : 第一幅图像上的特征描述子
     * @param desc2             : 第二幅图像上的特征描述子
     * @param matches           : [output] 第一幅图像和第二幅图像的特征匹配
     * @param distance_ratio    : 1NN和2NN之间的距离阈值
     */
    static
    void ComputeCrossMatches(const cv::Mat& desc1, const cv::Mat& desc2, std::vector<cv::DMatch>& matches, const float distance_ratio = 0.8);

    /**
     * 使用基础矩阵对特征匹配进行过滤
     * @param pts1          : 第一幅图像上的特征点
     * @param pts2          : 第二幅图像上的特征点
     * @param matches       : 第一幅图像和第二幅图像的特征匹配
     * @param prune_matches : [output] 使用RANSAC + 基础矩阵提纯之后的特征匹配
     */
    static
    void FilterMatches(const std::vector<cv::Point2f>& pts1,
                       const std::vector<cv::Point2f>& pts2,
                       const std::vector<cv::DMatch>& matches,
                       std::vector<cv::DMatch>& prune_matches);

    /**
     * 使用最大距离对匹配进行过滤，如果特征描述子之间的距离大于max_distance，那么认为不是正确的匹配.
     * 要求特征描述子已经进行了L1Root归一化或者是L2归一化
     * @param matches           :   待过滤的匹配
     * @param prune_matches     :   [output] 过滤后的匹配
     * @param max_distance      :   最大距离
     */
    static
    void FilterMatchesByDistance(const std::vector<cv::DMatch>& matches,
                                 std::vector<cv::DMatch>& prune_matches,
                                 const double& max_distance = 0.7);


    /**
     * 显示匹配
     * @param image1        : 第一幅图像
     * @param image2        : 第二幅图像
     * @param pts1          : 第一幅图像上的特征点
     * @param pts2          : 第二幅图像上的特征点
     * @param matches       : 第一幅图像和第二幅图像的特征匹配
     * @param window_name   : 显示图像窗口的名字
     * @param duration      : 持续时间
     */

    static
    void ShowMatches(const cv::Mat& image1,
                     const cv::Mat& image2,
                     const std::vector<cv::Point2f>& pts1,
                     const std::vector<cv::Point2f>& pts2,
                     const std::vector<cv::DMatch>& matches,
                     const std::string& window_name,
                     const time_t duration = 1000);


    /**
     * 显示匹配
     * @param image1        : 第一幅图像名字
     * @param image2        : 第二幅图像名字
     * @param pts1          : 第一幅图像上的特征点
     * @param pts2          : 第二幅图像上的特征点
     * @param matches       : 第一幅图像和第二幅图像的特征匹配
     * @param window_name   : 显示图像窗口的名字
     * @param duration      : 持续时间
     */
    static
    void ShowMatches(const std::string& image_name1,
                     const std::string& image_name2,
                     const std::vector<cv::Point2f>& pts1,
                     const std::vector<cv::Point2f>& pts2,
                     const std::vector<cv::DMatch>& matches,
                     const std::string& window_name,
                     const time_t duration = 1000);
    /**
     * 根据匹配关系，得到对齐之后的特征点
     * 即aligned_pts1[i]与aligned_pts2[i]是匹配点
     * @param pts1            : 第一幅图像上的特征点
     * @param pts2            : 第二幅图像上的特征点
     * @param matches         : 第一幅图像和第二幅图像的特征匹配
     * @param aligned_pts1    : [output] 第一幅图像对齐之后的特征点
     * @param aligned_pts2    : [output] 第二幅图像对齐之后的特征点
     */
    static
    void GetAlignedPointsFromMatches(const std::vector<cv::Point2f>& pts1,
                                     const std::vector<cv::Point2f>& pts2,
                                     const std::vector<cv::DMatch>& matches,
                                     std::vector<cv::Point2f>& aligned_pts1,
                                     std::vector<cv::Point2f>& aligned_pts2);
private:

    /**
     * 对匹配进行交叉验证， 得到通过交叉验证后的匹配关系
     * @param matches12     : 第一幅图像和第二幅图像之间的匹配
     * @param matches21     : 第二幅图像和第一幅图像之间的匹配
     * @param prune_matches : [output] 提出之后的第一幅图像与第二幅图像之间的匹配
     */
    static
    void CrossCheck(const std::vector<cv::DMatch>& matches12,
                    const std::vector<cv::DMatch>& matches21,
                    std::vector<cv::DMatch>& prune_matches);
};





} // namespace MonocularSfM


#endif // __FEATURE_UTILS_H__
