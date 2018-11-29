#ifndef __TRACK_H__
#define __TRACK_H__

#include <cstddef>
#include <cassert>
#include <vector>
#include <algorithm>

#include "Common/Types.h"

namespace MonocularSfM
{

struct TrackElement
{
    TrackElement(){}
    TrackElement(image_t image_id, point2D_t point2D_idx)
                  : image_id(image_id), point2D_idx(point2D_idx) {}

    image_t image_id;
    point2D_t point2D_idx;
};

class Track
{
public:


    ////////////////////////////////////////////////////////////////////////////////
    // 获得track的长度
    ////////////////////////////////////////////////////////////////////////////////
    size_t Length() const;

    ////////////////////////////////////////////////////////////////////////////////
    // 获取整个track
    ////////////////////////////////////////////////////////////////////////////////
    const std::vector<TrackElement>& Elements() const;
    std::vector<TrackElement>& Elements();


    ////////////////////////////////////////////////////////////////////////////////
    // 获取track中的元素
    ////////////////////////////////////////////////////////////////////////////////
    const TrackElement& Element(const size_t idx) const;
    TrackElement& Element(const size_t idx);

    ////////////////////////////////////////////////////////////////////////////////
    // 向track中添加元素
    ////////////////////////////////////////////////////////////////////////////////
    void AddElement(const TrackElement& element);
    void AddElement(const image_t image_id, const point2D_t point2D_idx);
    void AddElements(const std::vector<TrackElement>& elements);

    ////////////////////////////////////////////////////////////////////////////////
    // 从track中删除元素
    ////////////////////////////////////////////////////////////////////////////////
    void DeleteElement(const size_t idx);
    void DeleteElement(const TrackElement& element);
    void DeleteElement(const image_t image_id, const point2D_t point2D_idx);

private:

    std::vector<TrackElement> elements_;

};


} //namespace MonocularSfM

#endif //__TRACK_H__
