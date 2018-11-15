#include <iostream>
#include <unordered_set>

#include "Reconstruction/SceneGraph.h"

using namespace MonocularSfM;




void SceneGraph::Load(const cv::Ptr<Database> database, const size_t min_num_matches)
{
    //////////////////////////////////////////////////////////////////////////////
    // Load matches
    //////////////////////////////////////////////////////////////////////////////

    std::cout << "Loading matches..." << std::flush;
    const std::vector<std::pair<image_pair_t, std::vector<cv::DMatch>>> image_pairs
            = database->ReadAllMatches();

    std::cout << "Total image pairs : " << image_pairs.size()<< std::endl;


    //////////////////////////////////////////////////////////////////////////////
    // Load images
    //////////////////////////////////////////////////////////////////////////////

    const std::vector<Database::Image> images = database->ReadAllImages();
    std::cout << "Total images : " << images.size() << std::endl;
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(images.size());

    for(const auto& image_pair : image_pairs)
    {
        if(image_pair.second.size() >= min_num_matches)
        {
            image_t image_id1;
            image_t image_id2;
            Database::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
            connected_image_ids.insert(image_id1);
            connected_image_ids.insert(image_id2);
        }
    }

    //////////////////////////////////////////////////////////////////////////////
    // Build scene graph
    //////////////////////////////////////////////////////////////////////////////
    std::cout << "Building scene graph..." << std::flush;

    /// 添加图片
    images_.reserve(connected_image_ids.size());
    for(const auto& image : images)
    {
        if(connected_image_ids.count(image.id) > 0)
        {
            point2D_t num_points2D = database->NumKeyPoints(image.id);
            AddImage(image.id, num_points2D);
        }
    }

    /// 添加匹配
    size_t num_ignored_image_pairs  = 0;
    for(const auto& image_pair : image_pairs)
    {
        if(image_pair.second.size() >= min_num_matches)
        {
            image_t image_id1;
            image_t image_id2;
            Database::PairIdToImagePair(image_pair.first, &image_id1, &image_id2);
            AddCorrespondences(image_id1, image_id2, image_pair.second);
        }
        else
        {
            num_ignored_image_pairs += 1;
        }
    }

    std::cout << "Total image pairs : " << image_pairs.size() << ".  Ignored : " << num_ignored_image_pairs << std::endl;
    Finalize();

}


void SceneGraph::Finalize()
{
    for (auto it = images_.begin(); it != images_.end();)
    {
        it->second.num_observations = 0;
        for (auto& corr : it->second.corrs)
        {
            corr.shrink_to_fit();
            if (corr.size() > 0)
            {
              it->second.num_observations += 1;
            }
        }

        if (it->second.num_observations == 0)
        {
            images_.erase(it++);
        }
        else
        {
            ++it;
        }
    }

}


size_t SceneGraph::NumImages() const
{
    return images_.size();
}


bool SceneGraph::ExistsImage(const image_t image_id) const
{
    return images_.count(image_id) > 0;
}


point2D_t SceneGraph::NumObservationsForImage(image_t image_id) const
{
    assert(ExistsImage(image_id));
    return images_.at(image_id).num_observations;
}


point2D_t SceneGraph::NumCorrespondencesForImage(image_t image_id) const
{
    assert(ExistsImage(image_id));
    return images_.at(image_id).num_correspondences;
}

point2D_t SceneGraph::NumCorrespondencesBetweenImages(const image_t image_id1, const image_t image_id2) const
{
    assert(ExistsImage(image_id1));
    assert(ExistsImage(image_id2));

    const image_pair_t pair_id = Database::ImagePairToPairId(image_id1, image_id2);
    if(image_pairs_.count(pair_id) == 0)
    {
        return 0;
    }
    else
    {
        return  image_pairs_.at(pair_id);
    }

}



void SceneGraph::AddImage(const image_t image_id, const size_t num_points2D)
{
    assert(!ExistsImage(image_id));
    images_[image_id].corrs.resize(num_points2D);
}


void SceneGraph::AddCorrespondences(const image_t image_id1, const image_t image_id2, const std::vector<cv::DMatch> &matches)
{
    if(image_id1 == image_id2)
    {
        fprintf(stderr, "WARNING : Cannot use self-matches for image_id = %d", image_id1);
        return;
    }

    assert(ExistsImage(image_id1));
    assert(ExistsImage(image_id2));


    struct Image& image1 = images_.at(image_id1);
    struct Image& image2 = images_.at(image_id2);

    image1.num_correspondences += matches.size();
    image2.num_correspondences += matches.size();

    const image_pair_t pair_id = Database::ImagePairToPairId(image_id1, image_id2);

    point2D_t& num_correspondences = image_pairs_[pair_id];
    num_correspondences += static_cast<point2D_t>(matches.size());

    for(size_t i = 0; i < matches.size(); ++i)
    {
        const point2D_t point2D_idx1 = matches[i].queryIdx;
        const point2D_t point2D_idx2 = matches[i].trainIdx;

        const bool valid_idx1 = point2D_idx1 < image1.corrs.size();
        const bool valid_idx2 = point2D_idx2 < image2.corrs.size();

        if(valid_idx1 && valid_idx2)
        {

            const bool duplicate =
                    std::find_if(image1.corrs[point2D_idx1].begin(),
                                 image1.corrs[point2D_idx1].end(),
                                 [image_id2, point2D_idx2](const Correspondence& corr)
                                 {
                                    return corr.image_id == image_id2 &&
                                            corr.point2D_idx == point2D_idx2;
                                 }) != image1.corrs[point2D_idx1].end();

            if(duplicate)
            {
                image1.num_correspondences -= 1;
                image2.num_correspondences -= 1;
                num_correspondences -= 1;
                fprintf(stderr, "WARNING : Duplicate correspondence between"
                                "point2D_idx = %d in image_id = %d and point2D_idx = %d in "
                                "image_id = %d\n", point2D_idx1, image_id1, point2D_idx2 ,image_id2);
            }
            else
            {
                std::vector<Correspondence>& corrs1 = image1.corrs[point2D_idx1];
                corrs1.emplace_back(image_id2, point2D_idx2);

                std::vector<Correspondence>& corrs2 = image2.corrs[point2D_idx2];
                corrs2.emplace_back(image_id1, point2D_idx1);
            }

        }
        else
        {
            image1.num_correspondences -= 1;
            image2.num_correspondences -= 1;
            num_correspondences -= 1;
            if(!valid_idx1)
            {
                fprintf(stderr, "WARNING : point2D_idx = %d in image_id = %d does not exist\n",
                        point2D_idx1, image_id1);
            }
            if(!valid_idx2)
            {
                fprintf(stderr, "WARNING : point2D_idx = %d in image_id = %d does not exist\n",
                        point2D_idx2, image_id2);
            }

        }
    }

}

const std::vector<typename SceneGraph::Correspondence>
SceneGraph::FindCorrespondences(const image_t image_id, const point2D_t point2D_idx) const
{
    assert(ExistsImage(image_id));
    return images_.at(image_id).corrs.at(point2D_idx);
}


std::vector<cv::DMatch> SceneGraph::FindCorrespondencesBetweenImages(const image_t image_id1, const image_t image_id2) const
{
    std::vector<cv::DMatch> found_corrs;
    const struct Image& image1 = images_.at(image_id1);
    for(point2D_t point2D_idx1 = 0; point2D_idx1 < image1.corrs.size(); ++point2D_idx1)
    {
        for(const Correspondence& corr1 : image1.corrs[point2D_idx1])
        {
            if(corr1.image_id == image_id2)
            {
                found_corrs.push_back(cv::DMatch(point2D_idx1, corr1.point2D_idx, 0));
            }
        }
    }
    return found_corrs;

}


bool SceneGraph::HasCorrespondences(const image_t image_id, const point2D_t point2D_idx) const
{
    return !images_.at(image_id).corrs.at(point2D_idx).empty();
}

bool SceneGraph::IsTwoViewObservation(const image_t image_id, const point2D_t point2D_idx) const
{
    const struct Image& image = images_.at(image_id);
    const std::vector<Correspondence>& corrs = image.corrs.at(point2D_idx);
    if(corrs.size() != 1)
    {
        return false;
    }

    const struct Image& other_image = images_.at(corrs[0].image_id);
    const std::vector<Correspondence>& other_corrs = other_image.corrs.at(corrs[0].point2D_idx);

    return other_corrs.size() == 1;
}

std::vector<image_t> SceneGraph::GetAllImageIds() const
{
    std::vector<image_t> image_ids;
    for(auto pair : images_)
    {
        image_ids.push_back(pair.first);
    }
    return image_ids;
}


const std::unordered_map<image_pair_t, point2D_t> SceneGraph::ImagePairs()
{
    return image_pairs_;
}

