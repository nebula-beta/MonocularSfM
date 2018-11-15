#ifndef __TRIANGULATION_H__
#define __TRIANGULATION_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <opencv2/opencv.hpp>

#include <vector>
namespace Eigen
{
    typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;

}

namespace MonocularSfM
{

cv::Point3f TriangulateMultiViewPoint(const cv::Mat& K,
                                      const std::vector<cv::Mat>& R,
                                      const std::vector<cv::Mat>& t,
                                      const std::vector<cv::Point2f>& point2Ds);



Eigen::Vector3d TriangulateMultiViewPoint(const std::vector<Eigen::Matrix3x4d>& proj_matrices,
                                          const std::vector<Eigen::Vector2d>& points) ;


Eigen::Vector3d TriangulateMultiViewPoint2(const std::vector<Eigen::Matrix3x4d> &proj_matrices,
                                           const std::vector<Eigen::Vector2d> &points);




cv::Point3f TriangulateMultiViewPoint(const std::vector<cv::Mat>& proj_matrices,
                                      const std::vector<cv::Point2f>& point2Ds);


cv::Point3f TriangulateMultiViewPoint2(const std::vector<cv::Mat>& proj_matrices,
                                       const std::vector<cv::Point2f>& point2Ds);


} // namespace MonocularSfM



#endif // __TRIANGULATION_H__
