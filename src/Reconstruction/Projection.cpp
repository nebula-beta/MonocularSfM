#include "Reconstruction/Projection.h"


using namespace MonocularSfM;

bool Projection::HasPositiveDepth(const cv::Vec3d& point3D,
                                  const cv::Mat& R,
                                  const cv::Mat& t)
{
    assert(R.type() == CV_64F);
    assert(t.type() == CV_64F);


    cv::Mat proj_matrix;
    cv::hconcat(R, t, proj_matrix);

    bool has_positive_depth = Projection::HasPositiveDepth(point3D, proj_matrix);
    return has_positive_depth;
}

bool Projection::HasPositiveDepth(const cv::Vec3d& point3D,
                                  const cv::Mat& R1,
                                  const cv::Mat& t1,
                                  const cv::Mat& R2,
                                  const cv::Mat& t2)
{
    assert(R1.type() == CV_64F);
    assert(t1.type() == CV_64F);
    assert(R2.type() == CV_64F);
    assert(t2.type() == CV_64F);

    cv::Mat proj_matrix1;
    cv::Mat proj_matrix2;
    cv::hconcat(R1, t1, proj_matrix1);
    cv::hconcat(R2, t2, proj_matrix2);

    bool has_positive_depth1 = Projection::HasPositiveDepth(point3D, proj_matrix1);
    bool has_positive_depth2 = Projection::HasPositiveDepth(point3D, proj_matrix2);

    return has_positive_depth1 && has_positive_depth2;
}

//proj_matrix = [R | t]
bool Projection::HasPositiveDepth(const cv::Vec3d &point3D,
                      const cv::Mat &proj_matrix)
{
    cv::Mat homo_point3D = cv::Mat::zeros(4, 1, CV_64F);
    homo_point3D.at<double>(0, 0) = point3D(0);
    homo_point3D.at<double>(1, 0) = point3D(1);
    homo_point3D.at<double>(2, 0) = point3D(2);
    homo_point3D.at<double>(3, 0) = 1;

    // 对3D点进行旋转+平移
    cv::Mat transform_point3D = proj_matrix * homo_point3D;
    bool has_positive_depth = transform_point3D.at<double>(2, 0) > std::numeric_limits<double>::epsilon();

    return has_positive_depth;
}

bool Projection::HasPositiveDepth(const cv::Mat &proj_matrix1,
                                  const cv::Mat &proj_matrix2,
                                  const cv::Vec3d &point3D)
{
    bool has_positive_depth1 = Projection::HasPositiveDepth(point3D, proj_matrix1);
    bool has_positive_depth2 = Projection::HasPositiveDepth(point3D, proj_matrix2);

    return has_positive_depth1 && has_positive_depth2;
}




double Projection::CalculateReprojectionError(const cv::Vec3d& point3D,
                                              const cv::Vec2d& point2D,
                                              const cv::Mat& R,
                                              const cv::Mat& t,
                                              const cv::Mat& K)
{
    assert(R.type() == CV_64F);
    assert(t.type() == CV_64F);
    assert(K.type() == CV_64F);

    cv::Mat proj_matrix;
    cv::hconcat(K * R, K * t, proj_matrix);
    double error = Projection::CalculateReprojectionError(point3D, point2D, proj_matrix);
    return error;
}

double Projection::CalculateReprojectionError(const cv::Vec3d& point3D,
                                              const cv::Vec2d& point2D1,
                                              const cv::Vec2d& point2D2,
                                              const cv::Mat& R1,
                                              const cv::Mat& t1,
                                              const cv::Mat& R2,
                                              const cv::Mat& t2,
                                              const cv::Mat& K)
{
    assert(R1.type() == CV_64F);
    assert(t1.type() == CV_64F);
    assert(R2.type() == CV_64F);
    assert(t2.type() == CV_64F);
    assert(K.type() == CV_64F);

    cv::Mat proj_matrix1;
    cv::Mat proj_matrix2;
    cv::hconcat(K * R1, K * t1, proj_matrix1);
    cv::hconcat(K * R2, K * t2, proj_matrix2);
    double error = Projection::CalculateReprojectionError(point3D, point2D1, point2D2, proj_matrix1, proj_matrix2);
    return error;

}

//proj_matrix = K[R | t]
double Projection::CalculateReprojectionError(const cv::Vec3d& point3D,
                                              const cv::Vec2d& point2D,
                                              const cv::Mat& proj_matrix)
{
    // 矩阵相乘 (3 x 4) (4 x 1) => (3 x 1)
    cv::Mat homo_point3D = (cv::Mat_<double>(4, 1) << point3D(0), point3D(1), point3D(2), 1);

    cv::Mat proj_point2D = proj_matrix * homo_point3D;


    proj_point2D /= proj_point2D.at<double>(2, 0);


    double diff_x = proj_point2D.at<double>(0, 0) - point2D(0);
    double diff_y = proj_point2D.at<double>(1, 0) - point2D(1);

    double error = std::sqrt(diff_x * diff_x + diff_y * diff_y);
    return error;

}

double Projection::CalculateReprojectionError(const cv::Vec3d& point3D,
                                              const cv::Vec2d& point2D1,
                                              const cv::Vec2d& point2D2,
                                              const cv::Mat& proj_matrix1,
                                              const cv::Mat& proj_matrix2)
{

    double error1 = Projection::CalculateReprojectionError(point3D, point2D1, proj_matrix1);
    double error2 = Projection::CalculateReprojectionError(point3D, point2D2, proj_matrix2);
    return (error1 + error2) / 2;
}



double Projection::CalculateParallaxAngle(const cv::Vec3d& point3D,
                                          const cv::Mat& R1,
                                          const cv::Mat& t1,
                                          const cv::Mat& R2,
                                          const cv::Mat& t2)
{
    assert(R1.type() == CV_64F);
    assert(t1.type() == CV_64F);
    assert(R2.type() == CV_64F);
    assert(t2.type() == CV_64F);

    // 光心在世界坐标系下的坐标
    cv::Mat O1 = -R1.t() * t1;
    cv::Mat O2 = -R2.t() * t2;

    cv::Vec3d proj_center1(O1.at<double>(0, 0), O1.at<double>(1, 0), O1.at<double>(2, 0));
    cv::Vec3d proj_center2(O2.at<double>(0, 0), O2.at<double>(1, 0), O2.at<double>(2, 0));

    double tri_angle = CalculateParallaxAngle(point3D, proj_center1, proj_center2);
    return tri_angle;
}

double Projection::CalculateParallaxAngle(const cv::Vec3d& point3d,
                                          const cv::Vec3d& proj_center1,
                                          const cv::Vec3d& proj_center2)
{
    // (1)余弦定理
    // cosA = (b^2 + c^2 - a^2) / 2bc
    // (2)也可以使用a * b = |a| * |b| * cos来计算
    const double baseline = norm(proj_center1 - proj_center2);
    const double ray1 = norm(point3d - proj_center1);
    const double ray2 = norm(point3d - proj_center2);

    const double angle = std::abs(
            std::acos((ray1 * ray1 + ray2 * ray2 - baseline * baseline) / (2 * ray1 * ray2))
    );

    if(std::isnan(angle))
    {
        return 0;
    }
    else
    {
        return std::min<double>(angle, M_PI - angle) * 180 / M_PI;
    }
}
