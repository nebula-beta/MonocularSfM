#ifndef __REGISTER_GRAPH_H__
#define __REGISTER_GRAPH_H__

#include <vector>
#include <cassert>
#include <cstddef>
#include <algorithm>

#include "Common/Types.h"

namespace MonocularSfM
{

class RegisterGraph
{
public:


public:
    RegisterGraph(const size_t& total_node);

    void AddEdge(const image_t& image_id1, const image_t& image_id2);


    bool IsRegistered(const image_t& image_id);
    void SetRegistered(const image_t& image_id);

    void AddNumTrial(const image_t& image_id);
    size_t GetNumTrial(const image_t& image_id);

    std::vector<size_t> GetAllImagesNumTrial();
    double GetMeanNumTrial();

    size_t NumRegisteredImage() const;

    // 得到下一次要注册的图片
    // 按照优先顺序从高到低进行排序
    std::vector<image_t> GetNextImageIds();


private:


    std::vector<std::vector<image_t>> nodes_;

    std::vector<bool> registered_;
    std::vector<size_t> num_registered_trials_;
    std::vector<size_t> num_registered_neighbor_;
    std::vector<size_t> registered_images_;
    size_t total_node_;
};

} // namespace MonocularSfM

#endif //__REGISTER_GRAPH_H__
