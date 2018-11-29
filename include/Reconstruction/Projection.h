#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include <opencv2/opencv.hpp>

namespace MonocularSfM
{


class Projection
{
public:

    static
    bool HasPositiveDepth(const cv::Vec3d& point3D,
                          const cv::Mat& R,
                          const cv::Mat& t);
    static
    bool HasPositiveDepth(const cv::Vec3d& point3D,
                          const cv::Mat& R1,
                          const cv::Mat& t1,
                          const cv::Mat& R2,
                          const cv::Mat& t2);

    //proj_matrix = [R | t]
    static
    bool HasPositiveDepth(const cv::Vec3d &point3D,
                          const cv::Mat &proj_matrix);
    // 冲突, 只能这样写
    static
    bool HasPositiveDepth(const cv::Mat &proj_matrix1,
                          const cv::Mat &proj_matrix2,
                          const cv::Vec3d &point3D);



    static
    double CalculateReprojectionError(const cv::Vec3d& point3D,
                                     const cv::Vec2d& point2D,
                                     const cv::Mat& R,
                                     const cv::Mat& t,
                                     const cv::Mat& K);
    static
    double CalculateReprojectionError(const cv::Vec3d& point3D,
                                     const cv::Vec2d& point2D1,
                                     const cv::Vec2d& point2D2,
                                     const cv::Mat& R1,
                                     const cv::Mat& t1,
                                     const cv::Mat& R2,
                                     const cv::Mat& t2,
                                     const cv::Mat& K);
    //proj_matrix = K[R | t]
    static
    double CalculateReprojectionError(const cv::Vec3d& point3D,
                                     const cv::Vec2d& point2D,
                                     const cv::Mat& proj_matrix);

    static
    double CalculateReprojectionError(const cv::Vec3d& point3D,
                                     const cv::Vec2d& point2D1,
                                     const cv::Vec2d& point2D2,
                                     const cv::Mat& proj_matrix1,
                                     const cv::Mat& proj_matrix2);


    static
    double CalculateParallaxAngle(const cv::Vec3d& point3D,
                                 const cv::Mat& R1,
                                 const cv::Mat& t1,
                                 const cv::Mat& R2,
                                 const cv::Mat& t2);
    static
    double CalculateParallaxAngle(const cv::Vec3d& point3d,
                                 const cv::Vec3d& proj_center1,
                                 const cv::Vec3d& proj_center2);

};



} //namespace MonocularSfM

#endif //__POINT3D_H__
