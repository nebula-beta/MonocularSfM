
#include <algorithm>

#include "Reconstruction/Projection.h"
#include "Reconstruction/Initializer.h"
#include "Reconstruction/Utils.h"

using namespace MonocularSfM;





Initializer::Initializer(const Parameters& params, const cv::Mat& K)
                        : params_(params), K_(K)
{
    assert(K_.type() == CV_64F);
}


Initializer::Statistics
Initializer::Initialize(const std::vector<cv::Vec2d>& points2D1,
                        const std::vector<cv::Vec2d>& points2D2)
{





    cv::Mat H;
    cv::Mat F;
    std::vector<bool> inlier_mask_H;
    std::vector<bool>inlier_mask_F;
    size_t num_inliers_H;
    size_t num_inliers_F;;


    FindHomography(points2D1, points2D2, H, inlier_mask_H, num_inliers_H);
    FindFundanmental(points2D1, points2D2, F, inlier_mask_F, num_inliers_F);

    assert(H.type() == CV_64F);
    assert(F.type() == CV_64F);

    double H_F_ratio = static_cast<double>(num_inliers_H) / static_cast<double>(num_inliers_F);



    statistics_.is_succeed = false;
    statistics_.num_inliers_F = num_inliers_F;
    statistics_.num_inliers_H = num_inliers_H;
    statistics_.H_F_ratio = H_F_ratio;

    // 不管使用何种方法进行初始化, 内点数都要满足要求
    if(H_F_ratio < 0.7 && num_inliers_F >= params_.rel_pose_min_num_inlier)
    {
        // 使用基础矩阵进行初始化
        RecoverPoseFromFundanmental(F, points2D1, points2D2, inlier_mask_F);

    }
    else if(num_inliers_H >= params_.rel_pose_min_num_inlier)
    {
        // 使用单应矩阵进行初始化
        RecoverPoseFromHomography(H, points2D1, points2D2, inlier_mask_H);
    }
    else
    {
        statistics_.fail_reason = "Not sufficient inliers";
    }


    PrintStatistics(statistics_);

    return statistics_;
}

void Initializer::PrintStatistics(const Statistics& statistics)
{
    const size_t kWidth = 20;
    std::cout.flags(std::ios::left); //左对齐
    std::cout << std::endl;
    std::cout << "--------------- Initialize Summary Start ---------------" << std::endl;
    std::cout << std::setw(kWidth) << "Initialize status" << " : " << (statistics.is_succeed ? "true" : "false") << std::endl;
    std::cout << std::setw(kWidth) << "Initialize method" << " : " << statistics.method << std::endl;
    if(!statistics.is_succeed)
    {
    std::cout << std::setw(kWidth) << "Fail reason"       << " : " << statistics.fail_reason << std::endl;
    }
    std::cout << std::setw(kWidth) << "Num inliers H"     << " : " << statistics.num_inliers_H << std::endl;
    std::cout << std::setw(kWidth) << "Num inliers F"     << " : " << statistics.num_inliers_F << std::endl;
    std::cout << std::setw(kWidth) << "H F ratio"         << " : " << statistics.H_F_ratio << std::endl;
    std::cout << std::setw(kWidth) << "Num inliers"       << " : " << statistics.num_inliers << std::endl;
    std::cout << std::setw(kWidth) << "Median tri angle"  << " : " << statistics.median_tri_angle << std::endl;
    std::cout << std::setw(kWidth) << "Ave tri angle"     << " : " << statistics.ave_tri_angle << std::endl;
    std::cout << std::setw(kWidth) << "Ave residual"      << " : " << statistics.ave_residual << std::endl;
    std::cout << "--------------- Initialize Summary End ---------------" << std::endl;
    std::cout << std::endl;
}





void Initializer::FindHomography(const std::vector<cv::Vec2d>& points2D1,
                                 const std::vector<cv::Vec2d>& points2D2,
                                 cv::Mat& H,
                                 std::vector<bool>& inlier_mask,
                                 size_t& num_inliers)
{
    cv::Mat cv_inlier_mask;
    H = cv::findHomography(Utils::Vector2dToPoint2f(points2D1),
                           Utils::Vector2dToPoint2f(points2D2),
                           cv::RANSAC, params_.rel_pose_homography_error,
                           cv_inlier_mask, 10000,
                           params_.rel_pose_ransac_confidence);


    assert(cv_inlier_mask.type() == CV_8U);
    inlier_mask.resize(cv_inlier_mask.rows, false);
    num_inliers = 0;

    for(int i = 0; i < cv_inlier_mask.rows; ++i)
    {
        if(cv_inlier_mask.at<uchar>(i, 0) == 0)
            continue;

        inlier_mask[i] = true;
        num_inliers += 1;
    }
}

void Initializer::FindFundanmental(const std::vector<cv::Vec2d>& points2D1,
                                   const std::vector<cv::Vec2d>& points2D2,
                                   cv::Mat& F,
                                   std::vector<bool>& inlier_mask,
                                   size_t& num_inliers)
{
    cv::Mat cv_inlier_mask;


    F = cv::findFundamentalMat(Utils::Vector2dToPoint2f(points2D1),
                               Utils::Vector2dToPoint2f(points2D2),
                               cv::FM_RANSAC, params_.rel_pose_essential_error,
                               params_.rel_pose_ransac_confidence, cv_inlier_mask);

    assert(cv_inlier_mask.type() == CV_8U);

    inlier_mask.resize(cv_inlier_mask.rows, false);
    num_inliers = 0;

    for(int i = 0; i < cv_inlier_mask.rows; ++i)
    {
        if(cv_inlier_mask.at<uchar>(i, 0) == 0)
            continue;

        inlier_mask[i] = true;
        num_inliers += 1;
    }

}








bool Initializer::RecoverPoseFromHomography(const cv::Mat& H,
                                            const std::vector<cv::Vec2d>& points2D1,
                                            const std::vector<cv::Vec2d>& points2D2,
                                            const std::vector<bool>& inlier_mask_H)
{

    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> ts;

    cv::decomposeHomographyMat(H, K_, Rs, ts, cv::noArray());

    size_t best_num_inlier = 0;


    // 在这些解中找到最合适的解
    for(size_t k = 0; k < Rs.size(); ++k)
    {


        cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat R2 = Rs[k];
        cv::Mat t2 = ts[k];


        cv::Mat P1, P2;

        cv::hconcat(K_ * R1, K_ * t1, P1);
        cv::hconcat(K_ * R2, K_ * t2, P2);




        std::vector<cv::Vec3d> points3D(points2D1.size());
        std::vector<double> tri_angles(points2D1.size(), 0);
        std::vector<double> residuals(points2D1.size(), std::numeric_limits<double>::max());
        std::vector<bool> inlier_mask(points2D1.size(), false);

        size_t num_inliers = 0;
        double sum_residual = 0;
        double sum_tri_angle = 0;


        for(size_t i = 0; i < points2D1.size(); ++i)
        {
            cv::Vec3d point3D = Triangulate(P1, P2, points2D1[i], points2D2[i]);



            bool has_positive_depth = Projection::HasPositiveDepth(point3D, R1, t1, R2, t2);
            double error = Projection::CalculateReprojectionError(point3D, points2D1[i], points2D2[i],
                                                          R1, t1, R2, t2, K_);
            double tri_angle = Projection::CalculateParallaxAngle(point3D, R1, t1, R2, t2);



            points3D[i] = point3D;
            tri_angles[i] = tri_angle;
            residuals[i] = error;


            if(has_positive_depth && error < params_.init_tri_max_error)
            {

                num_inliers  += 1;
                inlier_mask[i] = true;
                sum_residual += error;
                sum_tri_angle += tri_angle;

            }
            else
            {
                inlier_mask[i] = false;
            }
        }
        if(num_inliers > best_num_inlier)
        {


            std::sort(tri_angles.begin(), tri_angles.end());

            double ave_tri_angle = sum_tri_angle / num_inliers;
            double ave_residual = sum_residual / num_inliers;
            double median_tri_angle = 0.0f;

            if(tri_angles.size() % 2 == 1)
            {
                median_tri_angle = tri_angles[tri_angles.size() / 2];
            }
            else
            {
                median_tri_angle = tri_angles[(tri_angles.size() - 1) / 2] + tri_angles[tri_angles.size() / 2];
                median_tri_angle /= 2;
            }


            best_num_inlier = num_inliers;
            statistics_.method = "Homography";
            statistics_.num_inliers = num_inliers;
            statistics_.median_tri_angle = median_tri_angle;
            statistics_.ave_tri_angle = ave_tri_angle;
            statistics_.ave_residual = ave_residual;
            statistics_.R1 = R1;
            statistics_.t1 = t1;
            statistics_.R2 = R2;
            statistics_.t2 = t2;
            statistics_.points3D = std::move(points3D);
            statistics_.tri_angles = std::move(tri_angles);
            statistics_.residuals = std::move(residuals);
            statistics_.inlier_mask = std::move(inlier_mask);
        }
    }
    //判断是否初始化成功
    if(statistics_.num_inliers < params_.rel_pose_min_num_inlier ||
       statistics_.median_tri_angle < params_.init_tri_min_angle ||
       statistics_.ave_tri_angle < params_.init_tri_min_angle ||
       statistics_.ave_residual > params_.init_tri_max_error)
    {
        statistics_.is_succeed = false;
        statistics_.fail_reason = GetFailReason();
    }
    else
    {
        statistics_.is_succeed = true;
    }

    return statistics_.is_succeed;

}

bool Initializer::RecoverPoseFromFundanmental(const cv::Mat& F,
                                              const std::vector<cv::Vec2d>& points2D1,
                                              const std::vector<cv::Vec2d>& points2D2,
                                              const std::vector<bool>& inlier_mask_F)
{
    cv::Mat E, R1, t1, R2, t2;
    cv::Mat inlier;

    // 由于使用 E = K_.t() * F * K_
    // 然后recoverPose会出错
    // 所以直接使用opencv的findEssentialMat
    // 然后再recoverPose
    E = cv::findEssentialMat(Utils::Vector2dToPoint2f(points2D1),
                             Utils::Vector2dToPoint2f(points2D2),
                             K_, cv::RANSAC,
                             params_.rel_pose_ransac_confidence,
                             params_.rel_pose_essential_error, inlier);

    cv::recoverPose(E, points2D1, points2D2, K_, R2, t2);


    R1 = cv::Mat::eye(3, 3, CV_64F);
    t1 = cv::Mat::zeros(3, 1, CV_64F);



    cv::Mat P1, P2;
    cv::hconcat(K_ * R1, K_ * t1, P1);
    cv::hconcat(K_ * R2, K_ * t2, P2);


    std::vector<cv::Vec3d> points3D(points2D1.size());
    std::vector<double> tri_angles(points2D1.size(), 0);
    std::vector<double> residuals(points2D1.size(), std::numeric_limits<double>::max());
    std::vector<bool> inlier_mask(points2D1.size(), false);

    size_t num_inliers = 0;
    double sum_residual = 0.0;
    double sum_tri_angle = 0.0;


    for(size_t i = 0; i < points2D1.size(); ++i)
    {
        if(inlier.at<uchar>(i, 0) == 0)
            continue;

        if(!inlier_mask_F[i])
            continue;

        cv::Vec3d point3D = Triangulate(P1, P2, points2D1[i], points2D2[i]);

        // 三角测量出来的点，要满足
        // 正深度
        // 重投影误差小于阈值
        // 三角测量的角度大于阈值

        bool has_positive_depth = Projection::HasPositiveDepth(point3D, R1, t1, R2, t2);
        double error = Projection::CalculateReprojectionError(point3D, points2D1[i], points2D2[i],
                                                              R1, t1, R2, t2, K_);

        double tri_angle = Projection::CalculateParallaxAngle(point3D, R1, t1, R2, t2);


        points3D[i] = point3D;
        tri_angles[i] = tri_angle;
        residuals[i] = error;


        if(has_positive_depth && error < params_.init_tri_max_error)
        {

            num_inliers  += 1;
            inlier_mask[i] = true;
            sum_residual += error;
            sum_tri_angle += tri_angle;

        }
        else
        {
            inlier_mask[i] = false;
        }
    }


    std::sort(tri_angles.begin(), tri_angles.end());

    double ave_tri_angle = sum_tri_angle / num_inliers;
    double ave_residual = sum_residual / num_inliers;
    double median_tri_angle = 0.0;

    // 获取已经成功三角测量（即重投影误差小于一定的阈值）的3D点的角度的中位数
    if(tri_angles.size() % 2 == 1)
    {
        median_tri_angle = tri_angles[tri_angles.size() / 2];
    }
    else
    {
        median_tri_angle = tri_angles[(tri_angles.size() - 1) / 2] + tri_angles[tri_angles.size() / 2];
        median_tri_angle /= 2;
    }


    //判断是否初始化成功
    if(num_inliers < params_.rel_pose_min_num_inlier ||
       median_tri_angle < params_.init_tri_min_angle ||
       ave_tri_angle < params_.init_tri_min_angle ||
       ave_residual > params_.init_tri_max_error)
    {
        statistics_.is_succeed = false;
        statistics_.fail_reason = GetFailReason();
        statistics_.num_inliers = num_inliers;
        statistics_.median_tri_angle = median_tri_angle;
        statistics_.ave_tri_angle = ave_tri_angle;
        statistics_.ave_residual = ave_residual;
        return false;
    }


    statistics_.is_succeed = true;
    statistics_.method = "Essential";
    statistics_.num_inliers = num_inliers;
    statistics_.median_tri_angle = median_tri_angle;
    statistics_.ave_tri_angle = ave_tri_angle;
    statistics_.ave_residual = ave_residual;
    statistics_.R1 = R1;
    statistics_.t1 = t1;
    statistics_.R2 = R2;
    statistics_.t2 = t2;
    statistics_.points3D = std::move(points3D);
    statistics_.tri_angles = std::move(tri_angles);
    statistics_.residuals = std::move(residuals);
    statistics_.inlier_mask = std::move(inlier_mask);

    return true;
}



cv::Vec3d Initializer::Triangulate(const cv::Mat& P1,
                                   const cv::Mat& P2,
                                   const cv::Vec2d& point2D1,
                                   const cv::Vec2d& point2D2)
{
    cv::Mat A(4, 4, CV_64F);
    // DLT
    A.row(0) = point2D1(0) * P1.row(2) - P1.row(0);
    A.row(1) = point2D1(1) * P1.row(2) - P1.row(1);
    A.row(2) = point2D2(0) * P2.row(2) - P2.row(0);
    A.row(3) = point2D2(1) * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u ,vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 最小特征值所对应的特征向量
    cv::Mat point3D = vt.row(3).t();

    assert(point3D.type() == CV_64F);

    // 从齐次坐标 -> 非齐次坐标
    point3D = point3D.rowRange(0, 3) / point3D.at<double>(3, 0);

    double x = point3D.at<double>(0);
    double y = point3D.at<double>(1);
    double z = point3D.at<double>(2);
    return cv::Vec3d(x, y, z);
}

std::string Initializer::GetFailReason()
{

    assert(!statistics_.is_succeed);

    std::string fail_reason = "";

    if(statistics_.num_inliers < params_.rel_pose_min_num_inlier)
    {
        fail_reason = "Not sufficient inliers";
    }
    if(statistics_.median_tri_angle < params_.init_tri_min_angle ||
       statistics_.ave_residual < params_.init_tri_min_angle)
    {
        fail_reason = (fail_reason.size() == 0) ? ("Not sufficient angle") : (fail_reason + " & Not sufficient angle");
    }
    if(statistics_.ave_residual > params_.init_tri_max_error)
    {
        fail_reason = (fail_reason.size() == 0) ? ("Too large triangulation error") : (fail_reason + " & Too large triangulation error");
    }

    return  fail_reason;
}

