#include "Reconstruction/Triangulation.h"
#include "Reconstruction/Filter.h"


namespace MonocularSfM
{


Eigen::Matrix3x4d ToMatrix3x4d(const cv::Mat &P)
{
    Eigen::Matrix<double,3,4> M;

    M << P.at<float>(0,0), P.at<float>(0,1), P.at<float>(0,2), P.at<float>(0, 3),
         P.at<float>(1,0), P.at<float>(1,1), P.at<float>(1,2), P.at<float>(1, 3),
         P.at<float>(2,0), P.at<float>(2,1), P.at<float>(2,2), P.at<float>(2, 3);


    return M;
}

cv::Point2f FromCameraToWorld(const cv::Mat& K, cv::Point2f point2D)
{
    const float f1 = K.at<float>(0, 0);
    const float f2 = K.at<float>(1, 1);
    const float c1 = K.at<float>(0 ,2);
    const float c2 = K.at<float>(1, 2);
    float u = (point2D.x - c1) / f1;
    float v = (point2D.y - c2) / f2;

    return cv::Point2f(u, v);
}

void FromCameraToWorld(const cv::Mat &K,
                       const std::vector<cv::Point2f>& camera_point2Ds,
                       std::vector<cv::Point2f>& world_point2Ds)
{
    world_point2Ds.reserve(camera_point2Ds.size());
    for(size_t i = 0; i < camera_point2Ds.size(); ++i)
    {
        world_point2Ds.push_back(FromCameraToWorld(K, camera_point2Ds[i]));
    }
}




cv::Point3f TriangulateMultiViewPoint2(const cv::Mat& K,
                                      const std::vector<cv::Mat>& R,
                                      const std::vector<cv::Mat>& t,
                                      const std::vector<cv::Point2f>& point2Ds)
{

    std::vector<cv::Mat> proj_matrices;

    for(size_t i = 0; i < R.size(); ++i)
    {
        cv::Mat P;
        cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
        cv::hconcat(I * R[i], I * t[i], P);
        proj_matrices.push_back(P);
    }
    std::vector<cv::Point2f> world_point2Ds;
    FromCameraToWorld(K, point2Ds, world_point2Ds);
    cv::Point3f point3D = TriangulateMultiViewPoint2(proj_matrices, world_point2Ds);

    return point3D;
}



cv::Point3f TriangulateMultiViewPoint(const cv::Mat& K,
                                      const std::vector<cv::Mat>& R,
                                      const std::vector<cv::Mat>& t,
                                      const std::vector<cv::Point2f>& point2Ds)

{
    std::vector<Eigen::Matrix3x4d> proj_matrices;
    std::vector<Eigen::Vector2d> points;

    for(size_t i = 0; i < R.size(); ++i)
    {
        cv::Mat P;
        cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
        cv::hconcat(I * R[i], I * t[i], P);
        proj_matrices.push_back(ToMatrix3x4d(P));
    }

    for(size_t i = 0; i < point2Ds.size(); ++i)
    {
        const float f1 = K.at<float>(0, 0);
        const float f2 = K.at<float>(1, 1);
        const float c1 = K.at<float>(0 ,2);
        const float c2 = K.at<float>(1, 2);
        float u = (point2Ds[i].x - c1) / f1;
        float v = (point2Ds[i].y - c2) / f2;
        points.push_back(Eigen::Vector2d(u, v));
    }

    Eigen::Vector3d p3d = TriangulateMultiViewPoint(proj_matrices, points);
    Eigen::Vector3d p3d2 = TriangulateMultiViewPoint2(proj_matrices, points);

//    std::cout << "-----------------" << std::endl;
//    std::cout << p3d(0) << " " << p3d(1) << " " << p3d(2) <<std::endl;
//    std::cout << p3d2(0) << " " << p3d2(1) << " " << p3d2(2) <<std::endl;
//    std::cout << "-----------------" << std::endl;

    cv::Point3f point3D(p3d(0), p3d(1), p3d(2));
    cv::Point3f point3D2(p3d2(0), p3d2(1), p3d2(2));


//    for(size_t i = 0; i < point2Ds.size(); ++i)
//    {
//        double err1 = PointFilter::CalculateReprojectionError(point3D, point2Ds[i], R[i], t[i], K);
//        double err2 = PointFilter::CalculateReprojectionError(point3D2, point2Ds[i], R[i], t[i], K);
//        std::cout << err1 << " " << err2 << std::endl;
//    }


    return point3D2;
}


Eigen::Vector3d TriangulateMultiViewPoint(const std::vector<Eigen::Matrix3x4d>& proj_matrices,
                                          const std::vector<Eigen::Vector2d>& points)
{

    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

    for (size_t i = 0; i < points.size(); i++)
    {
        const Eigen::Vector3d point = points[i].homogeneous().normalized();
        const Eigen::Matrix3x4d term =
            proj_matrices[i] - point * point.transpose() * proj_matrices[i];
        A += term.transpose() * term;
    }

    // The algorithm exploits the fact that the matrix is selfadjoint,
    // making it faster and more accurate than the general purpose eigenvalue algorithms
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

    return eigen_solver.eigenvectors().col(0).hnormalized();
}

Eigen::Vector3d TriangulateMultiViewPoint2(const std::vector<Eigen::Matrix3x4d> &proj_matrices,
                                           const std::vector<Eigen::Vector2d> &points)
{

    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

    for(size_t i = 0; i < points.size(); ++i)
    {
        const Eigen::Matrix<double, 1, 4> term1 = points[i](0) * proj_matrices[i].row(2) -  proj_matrices[i].row(0);
        const Eigen::Matrix<double, 1, 4> term2 = points[i](1) * proj_matrices[i].row(2) -  proj_matrices[i].row(1);
        A += term1.transpose() * term1;
        A += term2.transpose() * term2;
    }
//    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
//    return svd.matrixV().col(3).hnormalized();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

    // hnormalized : 齐次坐标的归一化, 也就是将坐标的最后一维变为1
    return eigen_solver.eigenvectors().col(0).hnormalized();
}



cv::Point3f TriangulateMultiViewPoint(const std::vector<cv::Mat> &proj_matrices, const std::vector<cv::Point2f> &point2Ds)
{
    cv::Mat A = cv::Mat::zeros(4, 4, CV_32F);
    for(size_t i = 0; i < point2Ds.size(); ++i)
    {
        cv::Mat h_point2D = (cv::Mat_<float>(3, 1) << point2Ds[i].x, point2Ds[i].y, 1);
        cv::normalize(h_point2D, h_point2D);

        const cv::Mat term = proj_matrices[i] - h_point2D * h_point2D.t() * proj_matrices[i];

        A += term.t() * term;
    }
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    cv::eigen(A, eigenvalues, eigenvectors);

    cv::Mat point3D;
    cv::convertPointsFromHomogeneous(eigenvectors.row(3), point3D);
    return cv::Point3f(point3D.at<float>(0, 0), point3D.at<float>(1, 0), point3D.at<float>(2, 0));
}

cv::Point3f TriangulateMultiViewPoint2(const std::vector<cv::Mat> &proj_matrices, const std::vector<cv::Point2f> &point2Ds)
{
    cv::Mat A = cv::Mat::zeros(4, 4, CV_32F);
    for(size_t i = 0; i < point2Ds.size(); ++i)
    {
        const cv::Mat term1 = point2Ds[i].x * proj_matrices[i].row(2) - proj_matrices[i].row(0);
        const cv::Mat term2 = point2Ds[i].y * proj_matrices[i].row(2) - proj_matrices[i].row(1);

        A += term1.t() * term1;
        A += term2.t() * term2;
    }
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    cv::eigen(A, eigenvalues, eigenvectors);

    cv::Mat point3D;
    cv::convertPointsFromHomogeneous(eigenvectors.row(3), point3D);
    return cv::Point3f(point3D.at<float>(0, 0), point3D.at<float>(1, 0), point3D.at<float>(2, 0));
}





}  // namespace MonocularSfM
