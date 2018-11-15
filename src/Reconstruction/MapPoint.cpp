#include "Reconstruction/MapPoint.h"

using namespace MonocularSfM;



const point3D_t& MapPoint::Point3DIdx() const
{
    return point3D_idx_;
}
point3D_t& MapPoint::Point3DIdx()
{
    return point3D_idx_;
}

void MapPoint::SetPoint3DIdx(const point3D_t& point3D_idx)
{
    point3D_idx_ = point3D_idx;
}


////////////////////////////////////////////////////////////////////////////////
// 存取整个track的3D点
////////////////////////////////////////////////////////////////////////////////
const cv::Point3f& MapPoint::Point3D() const
{
    return point3D_;
}
cv::Point3f& MapPoint::Point3D()
{
    return point3D_;
}

void MapPoint::SetPoint3D(const cv::Point3f& point3D)
{
    point3D_ = point3D;
}



////////////////////////////////////////////////////////////////////////////////
// 获得track的长度
////////////////////////////////////////////////////////////////////////////////
size_t MapPoint::Length() const
{
    return elements_.size();
}



////////////////////////////////////////////////////////////////////////////////
// 获取整个track
////////////////////////////////////////////////////////////////////////////////
const std::vector<MapPointElement>& MapPoint::Elements() const
{
    return elements_;
}
std::vector<MapPointElement>& MapPoint::Elements()
{
    return elements_;
}


////////////////////////////////////////////////////////////////////////////////
// 获取track中的元素
////////////////////////////////////////////////////////////////////////////////
const MapPointElement& MapPoint::Element(const size_t idx) const
{
    assert(idx < elements_.size());
    return elements_[idx];
}
MapPointElement& MapPoint::Element(const size_t idx)
{
    assert(idx < elements_.size());
    return elements_[idx];
}


////////////////////////////////////////////////////////////////////////////////
// 向track中添加元素
////////////////////////////////////////////////////////////////////////////////
void MapPoint::AddElement(const MapPointElement& element)
{
    elements_.push_back(element);
}

void MapPoint::AddElement(const image_t image_id, const point2D_t point2D_idx)
{
    elements_.emplace_back(image_id, point2D_idx);
}

void MapPoint::AddElements(const std::vector<MapPointElement>& elements)
{
    elements_.insert(elements_.end(), elements.begin(), elements.end());
}





////////////////////////////////////////////////////////////////////////////////
// 从track中删除元素
////////////////////////////////////////////////////////////////////////////////
void MapPoint::DeleteElement(const size_t idx)
{
    assert(idx < elements_.size());
    elements_.erase(elements_.begin() + idx);
}

void MapPoint::DeleteElement(const MapPointElement& element)
{
    DeleteElement(element.image_id, element.point2D_idx);
}

void MapPoint::DeleteElement(const image_t image_id, const point2D_t point2D_idx)
{
    elements_.erase(
                std::remove_if(elements_.begin(), elements_.end(),
                [image_id, point2D_idx](const MapPointElement& element)
                {
                    return element.image_id == image_id &&
                            element.point2D_idx == point2D_idx;
                })
                , elements_.end());
}
