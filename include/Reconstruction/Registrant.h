#ifndef __REGISTRANT_H__
#define __REGISTRANT_H__

#include <cstddef>
#include <vector>
#include <opencv2/opencv.hpp>
namespace MonocularSfM
{

class Registrant
{
public:
    enum class PnPMethod
    {
        P3P = 0,
        AP3P = 1,
        EPNP = 2,
        UPNP = 3
    };
    struct Parameters
    {
        size_t abs_pose_min_num_inliers = 15;            // PnP之后，2D-3D对应的数量大于该阈值，才认为成功
        PnPMethod pnp_method = PnPMethod::EPNP;           // PnP方法
        size_t abs_pose_num_iterative_optimize = 10000;  //
        double abs_pose_ransac_confidence = 0.9999;      // PnP ransac置信度
        double abs_pose_max_error = 4.0;                 // PnP时，误差小于该阈值的点，被认为是内点

    };

    struct Statistics
    {
        bool is_succeed = false;
        size_t num_point2D_3D_correspondences = 0;
        size_t num_inliers = 0;

        double ave_residual = 0;

        cv::Mat R;
        cv::Mat t;
        std::vector<double> residuals;
        std::vector<bool> inlier_mask;

    };

public:
    Registrant(const Parameters& params, const cv::Mat& K);

    /**
     * 对于传进行来的3D-2D点对应， 使用PnP算法进行求解位姿
     */
    Statistics Register(const std::vector<cv::Vec3d>& points3D,
                        const std::vector<cv::Vec2d>& points2D);

    void PrintStatistics(const Statistics& statistics);
private:

    Parameters params_;
    Statistics statistics_;

    cv::Mat K_;
};


} //namespace MonocularSfM

#endif //__REGISTRANT_H__
