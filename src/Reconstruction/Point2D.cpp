#include "Reconstruction/Point2D.h"

using namespace MonocularSfM;

Point2D::Point2D() : xy_(0.0, 0.0), point3D_id_(INVALID)
{

}

Point2D::Point2D(const cv::Vec2d& xy, const cv::Vec3b& color, const point3D_t& point3D_id)
                : xy_(xy), color_(color), point3D_id_(point3D_id)
{

}

const cv::Vec2d& Point2D::XY() const
{
    return xy_;
}
cv::Vec2d Point2D::XY()
{
    return xy_;
}
void Point2D::SetXY(const cv::Vec2d& xy)
{
    xy_ = xy;
}


const cv::Vec3b& Point2D::Color() const
{
    return color_;
}

cv::Vec3b Point2D::Color()
{
    return color_;
}

void Point2D::SetColor(const cv::Vec3b& color)
{
    color_ = color;
}


const point3D_t& Point2D::Point3DId() const
{
    return point3D_id_;
}

point3D_t Point2D::Point3DId()
{
    return point3D_id_;
}
void Point2D::SetPoint3D(const point3D_t& point3D_id)
{
    point3D_id_ = point3D_id;
}

void Point2D::ResetPoint3D()
{
    point3D_id_ = INVALID;
}


bool Point2D::HasPoint3D() const
{
    return point3D_id_ != INVALID;
}
