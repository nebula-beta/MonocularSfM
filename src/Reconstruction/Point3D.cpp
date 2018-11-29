#include "Reconstruction/Point3D.h"
using namespace MonocularSfM;

Point3D::Point3D() : xyz_(0.0, 0.0, 0.0), color_(0, 0, 0), error_(-1.0)
{

}

Point3D::Point3D(const cv::Vec3d& xyz, const cv::Vec3b& color, const double& error)
                : xyz_(xyz), color_(color), error_(error)
{

}

const cv::Vec3d& Point3D::XYZ() const
{
    return xyz_;
}
cv::Vec3d Point3D::XYZ()
{
    return xyz_;
}

void Point3D::SetXYZ(const cv::Vec3d& xyz)
{
    xyz_ = xyz;
}

const cv::Vec3b Point3D::Color() const
{
    return color_;
}
cv::Vec3b Point3D::Color()
{
    return color_;
}

void Point3D::SetColor(const cv::Vec3b& color)
{
    color_ = color;
}

double Point3D::Error() const
{
    return error_;
}
bool Point3D::HasError() const
{
    // TODO 改善这个判断
    return (error_ != -1.0);
}
void Point3D::SetError(const double& error)
{
    error_ = error;
}

const class Track& Point3D::Track() const
{
    return track_;
}
class Track Point3D::Track()
{
    return track_;
}

void Point3D::SetTrack(const class Track& track)
{
    track_ = track;
}



void Point3D::AddElement(const TrackElement& element)
{
    track_.AddElement(element);
}
void Point3D::AddElement(const image_t image_id, const point2D_t point2D_idx)
{
    track_.AddElement(image_id, point2D_idx);
}
void Point3D::AddElements(const std::vector<TrackElement>& elements)
{
    track_.AddElements(elements);
}


void Point3D::DeleteElement(const size_t idx)
{
    track_.DeleteElement(idx);
}
void Point3D::DeleteElement(const TrackElement& element)
{
    track_.DeleteElement(element);
}
void Point3D::DeleteElement(const image_t image_id, const point2D_t point2D_idx)
{
    track_.DeleteElement(image_id, point2D_idx);
}
