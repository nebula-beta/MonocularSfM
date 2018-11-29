#include "Reconstruction/Triangulator.h"
#include "Reconstruction/Projection.h"

using namespace MonocularSfM;





Triangulator::Triangulator(const Parameters& params, const cv::Mat& K) : params_(params), K_(K)
{

}

Triangulator::Statistics
Triangulator::Triangulate(const std::vector<cv::Mat>& Rs,
                          const std::vector<cv::Mat>& ts,
                          const std::vector<cv::Vec2d>& points2D)
{

    assert(Rs.size() != 0);
    assert(Rs.size() == ts.size());
    assert(Rs.size() == points2D.size());


    cv::Vec3d point3D = TriangulateMultiviewPoint(Rs, ts, points2D);

    statistics_.is_succeed = false;
    statistics_.point3D = point3D;
    statistics_.ave_residual = 0;


    double sum_residual = 0;
    size_t num_inliers = 0;
    std::vector<double> residuals(points2D.size());

    // 检查重投影误差是否满足要求
    for(size_t i = 0; i < points2D.size(); ++i)
    {
        double error = Projection::CalculateReprojectionError(point3D, points2D[i], Rs[i], ts[i], K_);
        residuals[i] = error;
        if(error > params_.regis_tri_max_error)
        {
            break;
        }
        sum_residual += error;
        num_inliers += 1;
    }


    if(num_inliers == points2D.size())
    {
        bool is_keep_point = false;

        // 检查角度是否满足要求
        for(size_t i = 0; i < points2D.size(); ++i)
        {

            for(size_t j = 0; j < i; ++j)
            {
                double tri_angle = Projection::CalculateParallaxAngle(point3D, Rs[i], ts[i], Rs[j], ts[j]);
                if(tri_angle >= params_.regis_tri_min_angle)
                {
                    is_keep_point = true;
                    break;
                }
            }
            if(is_keep_point)
            {
                break;
            }
        }

        if(is_keep_point)
        {
            statistics_.is_succeed = true;
            statistics_.ave_residual = sum_residual / num_inliers;
            statistics_.residuals = std::move(residuals);
        }
    }

    return statistics_;
}



cv::Vec3d Triangulator::TriangulateMultiviewPoint(const std::vector<cv::Mat>& Rs,
                                                  const std::vector<cv::Mat>& ts,
                                                  const std::vector<cv::Vec2d>& points2D)
{
    cv::Mat A = cv::Mat::zeros(4, 4, CV_64F);
    for(size_t i = 0; i < points2D.size(); ++i)
    {
        cv::Mat proj_matrix;
        cv::hconcat(K_ * Rs[i], K_ * ts[i], proj_matrix);

        const cv::Mat term1 = points2D[i](0) * proj_matrix.row(2) - proj_matrix.row(0);
        const cv::Mat term2 = points2D[i](1) * proj_matrix.row(2) - proj_matrix.row(1);

        A += term1.t() * term1;
        A += term2.t() * term2;
    }

    cv::Mat eigenvalues;
    cv::Mat eigenvector;
    cv::eigen(A, eigenvalues, eigenvector);

    assert(eigenvector.type() == CV_64F);

    double x = eigenvector.at<double>(3, 0) / eigenvector.at<double>(3, 3);
    double y = eigenvector.at<double>(3, 1) / eigenvector.at<double>(3, 3);
    double z = eigenvector.at<double>(3, 2) / eigenvector.at<double>(3, 3);

    cv::Vec3d point3D(x, y, z);

    return point3D;
}
