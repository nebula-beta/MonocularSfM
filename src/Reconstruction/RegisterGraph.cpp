#include "Reconstruction/RegisterGraph.h"
using namespace MonocularSfM;

RegisterGraph::RegisterGraph(const size_t& total_node) : total_node_(total_node)
{
    nodes_ .resize(total_node_);
    registered_.resize(total_node_, false);
    num_registered_trials_.resize(total_node_, 0);
    num_registered_neighbor_.resize(total_node_, 0);

}




void RegisterGraph::AddEdge(const image_t &image_id1, const image_t &image_id2)
{
    assert(image_id1 < total_node_);
    assert(image_id2 < total_node_);
    assert(image_id1 != image_id2);

    nodes_[image_id1].push_back(image_id2);
    nodes_[image_id2].push_back(image_id1);
}


bool RegisterGraph::IsRegistered(const image_t& image_id)
{
    assert(image_id < total_node_);

    return registered_[image_id];
}

void RegisterGraph::SetRegistered(const image_t &image_id)
{
    registered_[image_id] = true;
    registered_images_.push_back(image_id);

    for(size_t i = 0; i < nodes_[image_id].size(); ++i)
    {
        image_t related_image_id = nodes_[image_id][i];
        num_registered_neighbor_[related_image_id] += 1;
    }
}

void RegisterGraph::AddNumTrial(const image_t& image_id)
{
    num_registered_trials_[image_id] += 1;
}

size_t RegisterGraph::GetNumTrial(const image_t& image_id)
{
    return num_registered_trials_[image_id];
}

std::vector<size_t> RegisterGraph::GetAllImagesNumTrial()
{
    return num_registered_trials_;
}
double RegisterGraph::GetMeanNumTrial()
{
    double sum_trials = 0;
    for(const size_t& num_trials : num_registered_trials_)
    {
        sum_trials += num_trials;
    }
    // TODO : 改善这里， 如果图片没有全注册， 那么分母是不正确的
    return sum_trials / num_registered_trials_.size();
}

size_t RegisterGraph::NumRegisteredImage() const
{
    return registered_images_.size();
}
std::vector<image_t> RegisterGraph::GetNextImageIds()
{
    struct ImageInfo
    {
        image_t image_id;
        double score;
    };

    std::vector<ImageInfo> good_bucket;
    std::vector<ImageInfo> bad_bucket;

    for(image_t image_id = 0; image_id < total_node_; ++image_id)
    {
        // 跳过已经注册的图片
        if(IsRegistered(image_id))
            continue;

        // 如果邻居没有注册, 那么也跳过
        if(num_registered_neighbor_[image_id] == 0)
            continue;

        ImageInfo image_info;
        image_info.image_id = image_id;
        image_info.score = num_registered_neighbor_[image_id];

        if(num_registered_trials_[image_id] == 0)
        {
            good_bucket.push_back(image_info);
        }
        else
        {
            bad_bucket.push_back(image_info);

        }
    }
    std::sort(
          good_bucket.begin(), good_bucket.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.score > image_info2.score;
            }
           );
    std::sort(
          bad_bucket.begin(), bad_bucket.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.score > image_info2.score;
            }
           );

    std::vector<image_t> image_ids;

    for(const ImageInfo& image_info : good_bucket)
    {
        image_ids.push_back(image_info.image_id);
    }

    for(const ImageInfo& image_info : bad_bucket)
    {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;

}
