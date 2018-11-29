#ifndef __INITIALIZER_H__
#define __INITIALIZER_H__


#include <cstddef>
#include <vector>
#include <opencv2/opencv.hpp>


namespace MonocularSfM
{

class Initializer
{
public:
    struct Parameters
    {

        size_t rel_pose_min_num_inlier = 100;        // 2D-2D点对应的内点数量的阈值

        double rel_pose_ransac_confidence = 0.9999;  // 求矩阵(H,E)时ransac的置信度

        double rel_pose_essential_error = 4.0;       // 求解决矩阵E的误差阈值

        double rel_pose_homography_error = 12.0;      // 求解决矩阵H的误差阈值


        double init_tri_max_error = 2.0;             // 三角测量时,重投影误差阈值

        double init_tri_min_angle = 4.0;             // 三角测量时, 角度阈值

    };
    struct Statistics
    {
        bool is_succeed = false;                     // 初始化是否成功
        std::string method = "None";                 // 初始化使用了何种方法
        std::string fail_reason = "None";

        size_t num_inliers_H = 0;                    // 估计单应矩阵时，符合单应矩阵的内点的数量
        size_t num_inliers_F = 0;                    // 估计基础矩阵时，符合基础矩阵的内点的数量
        double H_F_ratio = 0;                        // 单应矩阵的内点的数量 除以 基础矩阵的内点的数量

        size_t num_inliers = 0;                      // 成功三角测量的3D点数(重投影误差小于阈值)
        double median_tri_angle = 0;                 // 成功三角测量的3D点角度的中位数
        double ave_tri_angle = 0;                    // 成功三角测量的3D点角度的平均值
        double ave_residual = 0;                     // 平均重投影误差
        cv::Mat R1;                                  // 旋转矩阵1(单位矩阵)
        cv::Mat t1;                                  // 平移向量1(零向量)
        cv::Mat R2;                                  // 旋转矩阵2
        cv::Mat t2;                                  // 平移向量2
        std::vector<cv::Vec3d> points3D;             // 所有2D点所测量出来的3D点,包含了inlier和outlier
        std::vector<double> tri_angles;              // 每个3D点的角度
        std::vector<double> residuals;               // 每个3D点的重投影误差
        std::vector<bool> inlier_mask;               // 标记哪个3D点是内点

    };

public:
    Initializer(const Parameters& params, const cv::Mat& K);


    /**
     * 对于传进来两张图像已经对齐好的特征点，并尝试进行初始化
     * 返回初始化的统计信息
     */
    Statistics Initialize(const std::vector<cv::Vec2d>& points2D1,
                          const std::vector<cv::Vec2d>& points2D2);

    void PrintStatistics(const Statistics& statistics);

private:


    /**
     * 使用RANSAC寻找满足points2D1和points2D2这两组点集对应关系的单应矩阵H
     * @param points2D1     : 点集1
     * @param points2D2     : 点集2
     * @param H             : [output] 单应矩阵
     * @param inlier_mask   : [output] 标志哪个是内点
     * @param num_inliers   : [output] 有多少个内点
     */
    void FindHomography(const std::vector<cv::Vec2d>& points2D1,
                        const std::vector<cv::Vec2d>& points2D2,
                        cv::Mat& H,
                        std::vector<bool>& inlier_mask,
                        size_t& num_inliers);

    /**
     * 使用RANSAC寻找满足points2D1和points2D2这两组点集对应关系的基础矩阵F
     * @param points2D1     : 点集1
     * @param points2D2     : 点集2
     * @param H             : [output] 基础矩阵
     * @param inlier_mask   : [output] 标志哪个是内点
     * @param num_inliers   : [output] 有多少个内点
     */
    void FindFundanmental(const std::vector<cv::Vec2d>& points2D1,
                          const std::vector<cv::Vec2d>& points2D2,
                          cv::Mat& F,
                          std::vector<bool>& inlier_mask,
                          size_t& num_inliers);


    /**
     * 分解单应矩阵H，从而得到初始位姿， 并进行三角测量，存到statistics_中
     * @param H                 : 单应矩阵
     * @param points2D1         : 点集1
     * @param points2D2         : 点集2
     * @param inlier_mask_H     : 调用FindHomography时，得到的inlier_mask
     * @return                  : true, 初始化成功； false,失败
     */
    bool RecoverPoseFromHomography(const cv::Mat& H,
                                   const std::vector<cv::Vec2d>& points2D1,
                                   const std::vector<cv::Vec2d>& points2D2,
                                   const std::vector<bool>& inlier_mask_H);

    /**
     * 分解基础矩阵F(实际上是分解本质矩阵E)，从而得到初始位姿， 并进行三角测量，存到statistics_中
     * @param F                 : 单应矩阵
     * @param points2D1         : 点集1
     * @param points2D2         : 点集2
     * @param inlier_mask_F     : 调用FindFundanmental时，得到的inlier_mask
     * @return                  : true, 初始化成功； false,失败
     */
    bool RecoverPoseFromFundanmental(const cv::Mat& F,
                                     const std::vector<cv::Vec2d>& points2D1,
                                     const std::vector<cv::Vec2d>& points2D2,
                                     const std::vector<bool>& inlier_mask_F);




    cv::Vec3d Triangulate(const cv::Mat& P1,
                          const cv::Mat& P2,
                          const cv::Vec2d& point2D1,
                          const cv::Vec2d& point2D2);

    /**
     * 根据statistics_， 得到初始化失败的信息
     * [note] : 只有当statistics_.is_succeed == false
     *          并且statistics_.num_inliers
     *             statistics_.median_tri_angle
     *             statistics_.ave_tri_angle
     *             statistics_.ave_residual
     *          不为空时，才能调用这个函数
     */
    std::string GetFailReason();


//    bool CheckCheirality(std::vector<cv::Vec3d>& points3D,
//                         const std::vector<cv::Vec2d>& points2D1,
//                         const std::vector<cv::Vec2d>& points2D2,
//                         const cv::Mat& R1,
//                         const cv::Mat& t1,
//                         const cv::Mat& R2,
//                         const cv::Mat& t2);


private:

    Parameters params_;
    Statistics statistics_;

    cv::Mat K_;


};


} //namespace MonocularSfM

#endif //__INITIALIZER_H__
