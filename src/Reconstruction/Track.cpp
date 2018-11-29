#include "Reconstruction/Track.h"


using namespace MonocularSfM;






////////////////////////////////////////////////////////////////////////////////
// 获得track的长度
////////////////////////////////////////////////////////////////////////////////
size_t Track::Length() const
{
    return elements_.size();
}



////////////////////////////////////////////////////////////////////////////////
// 获取整个track
////////////////////////////////////////////////////////////////////////////////
const std::vector<TrackElement>& Track::Elements() const
{
    return elements_;
}
std::vector<TrackElement>& Track::Elements()
{
    return elements_;
}


////////////////////////////////////////////////////////////////////////////////
// 获取track中的元素
////////////////////////////////////////////////////////////////////////////////
const TrackElement& Track::Element(const size_t idx) const
{
    assert(idx < elements_.size());
    return elements_[idx];
}
TrackElement& Track::Element(const size_t idx)
{
    assert(idx < elements_.size());
    return elements_[idx];
}


////////////////////////////////////////////////////////////////////////////////
// 向track中添加元素
////////////////////////////////////////////////////////////////////////////////
void Track::AddElement(const TrackElement& element)
{
    elements_.push_back(element);
}

void Track::AddElement(const image_t image_id, const point2D_t point2D_idx)
{
    elements_.emplace_back(image_id, point2D_idx);
}

void Track::AddElements(const std::vector<TrackElement>& elements)
{
    elements_.insert(elements_.end(), elements.begin(), elements.end());
}





////////////////////////////////////////////////////////////////////////////////
// 从track中删除元素
////////////////////////////////////////////////////////////////////////////////
void Track::DeleteElement(const size_t idx)
{
    assert(idx < elements_.size());
    elements_.erase(elements_.begin() + idx);
}

void Track::DeleteElement(const TrackElement& element)
{
    DeleteElement(element.image_id, element.point2D_idx);
}

void Track::DeleteElement(const image_t image_id, const point2D_t point2D_idx)
{
    elements_.erase(
                std::remove_if(elements_.begin(), elements_.end(),
                [image_id, point2D_idx](const TrackElement& element)
                {
                    return element.image_id == image_id &&
                            element.point2D_idx == point2D_idx;
                })
                , elements_.end());
}
