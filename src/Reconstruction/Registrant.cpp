#include "Reconstruction/Registrant.h"
#include "Reconstruction/Utils.h"
#include "Reconstruction/Projection.h"

using namespace MonocularSfM;

Registrant::Registrant(const Parameters& params, const cv::Mat& K)
                      : params_(params), K_(K)
{
    assert(K_.type() == CV_64F);
}

Registrant::Statistics Registrant::Register(const std::vector<cv::Vec3d> &points3D,
                                            const std::vector<cv::Vec2d> &points2D)
{
    assert(points2D.size() == points3D.size());

    statistics_.is_succeed = false;
    statistics_.num_point2D_3D_correspondences = points2D.size();
    statistics_.num_inliers = 0;
    statistics_.ave_residual = 0;

    if(points2D.size() < params_.abs_pose_min_num_inliers)
    {
        return statistics_;
    }
    cv::Mat inlier_idxs;
    cv::Mat rvec;
    cv::Mat t;
    /// PnP求解位姿
    /// 需要注意的是solvePnPRansac的inlier_idxs输出
    /// inlier_idx = inlier_idxs[i] 表示inlier_idx个点是内点，
    /// 此时的mask的值不再是0和1
    bool kUseExtrinsicGuess = false;

    // TODO : 弄清除solvePnPRansac的运行时间和abs_pose_num_iterative_optimize有没有关系
    // TODO : 还是只有选择使用cv::SOLVEPNP_ITERATIVE时， 参数abs_pose_num_iterative_optimize才起作用
    switch (params_.pnp_method)
    {
        case PnPMethod::P3P:
            cv::solvePnPRansac(Utils::Vector3dToPoint3f(points3D), Utils::Vector2dToPoint2f(points2D),
                               K_, cv::Mat(), rvec, t, kUseExtrinsicGuess,
                               params_.abs_pose_num_iterative_optimize, params_.abs_pose_max_error,
                               params_.abs_pose_ransac_confidence, inlier_idxs, cv::SOLVEPNP_P3P);
            break;
        case PnPMethod::AP3P:
            cv::solvePnPRansac(Utils::Vector3dToPoint3f(points3D), Utils::Vector2dToPoint2f(points2D),
                               K_, cv::Mat(), rvec, t, kUseExtrinsicGuess,
                               params_.abs_pose_num_iterative_optimize, params_.abs_pose_max_error,
                               params_.abs_pose_ransac_confidence, inlier_idxs, cv::SOLVEPNP_AP3P);
        break;
        case PnPMethod::EPNP:
            cv::solvePnPRansac(Utils::Vector3dToPoint3f(points3D), Utils::Vector2dToPoint2f(points2D),
                               K_, cv::Mat(), rvec, t, kUseExtrinsicGuess,
                               params_.abs_pose_num_iterative_optimize, params_.abs_pose_max_error,
                               params_.abs_pose_ransac_confidence, inlier_idxs, cv::SOLVEPNP_UPNP);
            break;
        case PnPMethod::UPNP:
            cv::solvePnPRansac(Utils::Vector3dToPoint3f(points3D), Utils::Vector2dToPoint2f(points2D),
                               K_, cv::Mat(), rvec, t, kUseExtrinsicGuess,
                               params_.abs_pose_num_iterative_optimize, params_.abs_pose_max_error,
                               params_.abs_pose_ransac_confidence, inlier_idxs, cv::SOLVEPNP_UPNP);
            break;
        default:
        assert(false);
    }





    // 这里的inlier_idxs.rows并不代表真正的2D-3D inlier的数量
    // 这是因为由于一个2D点有多个匹配点， 这些匹配点可能对应着不同的3D点
    // 从而导致2D-2D-3D传递时， 得到同一个2D点对应多个3D点
    // 所以如果inlier_idxs中存在多个相同的2D点， 那么只能保留一个
    // 所以inlier_idxs.rows并不代表真正的2D-3D inlier的数量
    // TODO : 改善： 得到真正的inlier的数量
    if(inlier_idxs.rows < params_.abs_pose_min_num_inliers)
    {
        return statistics_;
    }

    std::vector<bool> inlier_mask(points2D.size(), false);

    for(int i = 0; i < inlier_idxs.rows; ++i)
    {
        int idx = inlier_idxs.at<int>(i, 0);
        inlier_mask[idx] = true;
    }

    assert(rvec.type() == CV_64F);
    assert(t.type() == CV_64F);



    cv::Mat R;
    cv::Rodrigues(rvec, R);


    double sum_residual = 0;
    std::vector<double> residuals(inlier_mask.size());
    for(size_t i = 0; i < inlier_mask.size(); ++i)
    {
        double error = Projection::CalculateReprojectionError(points3D[i], points2D[i], R, t, K_);
        if(!inlier_mask[i])
            continue;
        sum_residual += error;
        residuals[i] = error;

    }



    statistics_.is_succeed = true;
    statistics_.num_inliers = static_cast<size_t>(inlier_idxs.rows);
    statistics_.ave_residual = sum_residual / statistics_.num_inliers;

    statistics_.R = R;
    statistics_.t = t;
    statistics_.residuals = std::move(residuals);
    statistics_.inlier_mask = std::move(inlier_mask);

    return statistics_;

}


void Registrant::PrintStatistics(const Statistics& statistics)
{
    const size_t kWidth = 30;
    std::cout.flags(std::ios::left); //左对齐
    std::cout << std::endl;
    std::cout << "--------------- Register Summary Start ---------------" << std::endl;
    std::cout << std::setw(kWidth) << "Initialize status"          << " : " << (statistics.is_succeed ? "true" : "false") << std::endl;
    std::cout << std::setw(kWidth) << "Num 2D 3D correspondences"  << " : " << statistics.num_point2D_3D_correspondences << std::endl;
    std::cout << std::setw(kWidth) << "Num inliers "               << " : " << statistics.num_inliers << std::endl;
    std::cout << std::setw(kWidth) << "Ave residual "              << " : " << statistics.ave_residual << std::endl;
    std::cout << "--------------- Register Summary End ---------------" << std::endl;
    std::cout << std::endl;
}
