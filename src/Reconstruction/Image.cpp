#include "Reconstruction/Image.h"


using namespace MonocularSfM;


image_t Image::ImageId() const
{
    return image_id_;
}
void Image::SetImageId(const image_t image_id)
{
    image_id_ = image_id;
}

const std::string& Image::Name() const
{
    return name_;
}
std::string& Image::Name()
{
    return name_;
}

void Image::SetName(const std::string& name)
{
    name_ = name;
}


const cv::Mat& Image::R() const
{
    return R_;
}
cv::Mat& Image::R()
{
    return R_;
}

void Image::SetR(const cv::Mat& R)
{
    R_ = R;
}

const cv::Mat& Image::t() const
{
    return t_;
}
cv::Mat& Image::t()
{
    return t_;
}

void Image::SetT(const cv::Mat& t)
{
    t_ = t;
}


const cv::Point2f& Image::Point2D(const point2D_t point2D_idx) const
{
    assert(point2D_idx < point2D_.size());
    return point2D_[point2D_idx];
}
cv::Point2f& Image::Point2D(const point2D_t point2D_idx)
{
    assert(point2D_idx < point2D_.size());
    return point2D_[point2D_idx];
}

const std::vector<cv::Point2f>& Image::Point2Ds() const
{
    return point2D_;
}
std::vector<cv::Point2f>& Image::Point2Ds()
{
    return point2D_;
}

point2D_t Image::NumPoints2D() const
{
    return point2D_.size();
}

const cv::Vec3b& Image::Color(const point2D_t point2D_idx) const
{

    assert(point2D_idx < colors_.size());
    return colors_[point2D_idx];
}
cv::Vec3b& Image::Color(const point2D_t point2D_idx)
{

    assert(point2D_idx < colors_.size());
    return colors_[point2D_idx];
}

const std::vector<cv::Vec3b>& Image::Colors() const
{
    return colors_;
}
std::vector<cv::Vec3b>& Image::Colors()
{
    return colors_;
}


void Image::SetPoint2D3DCorrespondence(const point2D_t point2D_idx, const point2D_t point3D_idx)
{
    assert(point2D_idx < point2D_.size());
    point2D_3D_corrs_[point2D_idx] = point3D_idx;
}

void Image::DisablePoint2D3DCorrespondence(const point2D_t point2D_idx)
{
    assert(IsPoint2DHasPoint3D(point2D_idx));

    point2D_3D_corrs_.erase(point2D_idx);

}

bool Image::IsPoint2DHasPoint3D(const point2D_t point2D_idx) const
{
    assert(point2D_idx < point2D_.size());
    return point2D_3D_corrs_.find(point2D_idx) != point2D_3D_corrs_.end();
}

point3D_t Image::GetPoint2D3DCorrespondence(const point2D_t point2D_idx)
{
    assert(point2D_idx < point2D_.size());
    assert(point2D_3D_corrs_.find(point2D_idx) != point2D_3D_corrs_.end());
    return point2D_3D_corrs_[point2D_idx];
}


size_t Image::NumPoint2D3DCorrespondence() const
{
    return point2D_3D_corrs_.size();
}


void Image::SetPointVisable(const point2D_t& point2D_idx)
{
    is_visable_point3D_[point2D_idx] = true;
}

void Image::DisablePointVisable(const point2D_t& point2D_idx)
{
    if(is_visable_point3D_.count(point2D_idx) != 0)
        is_visable_point3D_.erase(point2D_idx);

}

size_t Image::NumVisablePoint3D() const
{
    return is_visable_point3D_.size();
}

