#include "Reconstruction/Utils.h"

using namespace MonocularSfM;



std::vector<cv::Point2f> Utils::Vector2dToPoint2f(const std::vector<cv::Vec2d>& points2D)
{
    std::vector<cv::Point2f> ret_points2D(points2D.size());

    for(size_t i = 0; i < points2D.size(); ++i)
    {
        float x = static_cast<float>(points2D[i](0));
        float y = static_cast<float>(points2D[i](1));

        ret_points2D[i] = cv::Point2f(x, y);
    }
    return ret_points2D;
}
std::vector<cv::Vec2d> Utils::Point2fToVector2d(const std::vector<cv::Point2f>& points2D)
{
    std::vector<cv::Vec2d> ret_points2D(points2D.size());

    for(size_t i = 0; i < points2D.size(); ++i)
    {
        double x = static_cast<double>(points2D[i].x);
        double y = static_cast<double>(points2D[i].y);

        ret_points2D[i] = cv::Vec2d(x, y);
    }
    return ret_points2D;
}


std::vector<cv::Point3f> Utils::Vector3dToPoint3f(const std::vector<cv::Vec3d>& points3D)
{
    std::vector<cv::Point3f> ret_points3D(points3D.size());

    for(size_t i = 0; i < points3D.size(); ++i)
    {
        float x = static_cast<float>(points3D[i](0));
        float y = static_cast<float>(points3D[i](1));
        float z = static_cast<float>(points3D[i](2));

        ret_points3D[i] = cv::Point3f(x, y, z);
    }
    return ret_points3D;
}

std::vector<cv::Vec3d> Utils::Point3fToVector3d(const std::vector<cv::Point3f>& points3D)
{
    std::vector<cv::Vec3d> ret_points3D(points3D.size());

    for(size_t i = 0; i < points3D.size(); ++i)
    {
        double x = static_cast<double>(points3D[i].x);
        double y = static_cast<double>(points3D[i].y);
        double z = static_cast<double>(points3D[i].z);

        ret_points3D[i] = cv::Vec3d(x, y, z);
    }
    return ret_points3D;
}



