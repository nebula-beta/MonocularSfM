#include "Common/Timer.h"
#include "Database/Database.h"
#include "Feature/FeatureUtils.h"
#include "Feature/FeatureMatching.h"


using namespace MonocularSfM;


void FeatureMatcher::MatchImagePairs(const std::vector<std::pair<image_t, image_t>>& image_pairs)
{

    database_->BeginTransaction();
    for(const auto& image_pair : image_pairs)
    {
        Timer timer;
        timer.Start();


        const image_t image_id1 = image_pair.first;
        const image_t image_id2 = image_pair.second;

        if(database_->ExistMatches(image_id1, image_id2))
        {
            std::cout << "Compute Matches " << image_id1 << " - " << image_id2 << " Existing, Continue!" << std::endl;
            continue;
        }

        std::cout << "Compute Matches " << image_id1 << " - " << image_id2 <<  " ... " << std::endl;

        // TODO : 优化, 进行cache
        cv::Mat desc1 = database_->ReadDescriptors(image_id1);
        cv::Mat desc2 = database_->ReadDescriptors(image_id2);
        std::vector<cv::DMatch> matches;

        if(cross_check_)
        {
            // 交叉验证，　你是我的匹配，　我也是你的匹配
            FeatureUtils::ComputeCrossMatches(desc1, desc2, matches, distance_ratio_);
        }
        else
        {
            FeatureUtils::ComputeMatches(desc1, desc2, matches, distance_ratio_);
        }


        std::vector<cv::DMatch> prune_matches;

        FeatureUtils::FilterMatchesByDistance(matches, prune_matches, max_distance_);


        std::vector<cv::KeyPoint> kpts1 = database_->ReadKeyPoints(image_id1);
        std::vector<cv::KeyPoint> kpts2 = database_->ReadKeyPoints(image_id2);
        std::vector<cv::Point2f> pts1, pts2;
        cv::KeyPoint::convert(kpts1, pts1);
        cv::KeyPoint::convert(kpts2, pts2);

        std::vector<cv::DMatch> geometric_verif_matches;
        //TODO : 只用了基础矩阵F来进行几何验证
        FeatureUtils::FilterMatches(pts1, pts2, prune_matches, geometric_verif_matches);


        std::cout <<"\t matches num : " << geometric_verif_matches.size() << std::endl;
        std::cout <<"\t ";
        timer.PrintSeconds();
        std::cout << std::endl;

//        const int kMinNumMatches = 15;
//        if(geometric_verif_matches.size() > kMinNumMatches)
        database_->WriteMatches(image_id1, image_id2, geometric_verif_matches);
    }
    database_->EndTransaction();
}

void SequentialFeatureMatcher::RunMatching()
{
    database_ = cv::Ptr<Database>(new Database());
    database_->Open(database_path_);

    std::vector<Database::Image> images = database_->ReadAllImages();

    for(size_t i = 1; i < images.size(); ++i)
    {
        std::vector<std::pair<image_t, image_t>> image_pairs;
        for(size_t k = 1; k <= overlap_; ++k)
        {
            // 因为 i - k 可能小于0, 如果使用size_t类型，会出错
            int j = static_cast<int>(i) - static_cast<int>(k);
            if(j < 0)
            {
                break;
            }
            image_pairs.push_back(std::make_pair<image_t, image_t>(i, static_cast<size_t>(j)));
        }

        MatchImagePairs(image_pairs);
    }
    database_->Close();

}

void BruteFeatureMatcher::RunMatching()
{
    database_ = cv::Ptr<Database>(new Database());
    database_->Open(database_path_);

    std::vector<Database::Image> images = database_->ReadAllImages();

    // TODO  实现抢占式匹配
    for(size_t i = 0; i < images.size(); ++i)
    {
        std::vector<std::pair<image_t, image_t>> image_pairs;
        int cur_paris_size = 0;
        for(size_t j = 0; j < i; ++j)
        {
            image_pairs.push_back(std::make_pair<image_t, image_t>(i, j));
            cur_paris_size += 1;
            if(cur_paris_size == max_pairs_size_)
            {
                if(is_preemtive_)
                {
                    image_pairs = PreemptivelyFilterImagePairs(image_pairs);
                }
                MatchImagePairs(image_pairs);
                image_pairs.clear();
                cur_paris_size = 0;
            }
        }

        if(cur_paris_size != 0)
        {

            if(is_preemtive_)
            {
                image_pairs = PreemptivelyFilterImagePairs(image_pairs);
            }

            MatchImagePairs(image_pairs);
            image_pairs.clear();
        }

    }
    database_->Close();

}


std::vector<std::pair<image_t, image_t>> BruteFeatureMatcher::PreemptivelyFilterImagePairs(std::vector<std::pair<image_t, image_t> > image_pairs)
{
    std::vector<std::pair<image_t, image_t>> filtered_image_pairs;


    for(const auto& image_pair : image_pairs)
    {
        const image_t image_id1 = image_pair.first;
        const image_t image_id2 = image_pair.second;


        cv::Mat top_scale_desc1 = GetTopScaleDescriptors(image_id1);
        cv::Mat top_scale_desc2 = GetTopScaleDescriptors(image_id2);

        std::vector<cv::DMatch> matches;
        if(cross_check_)
        {
            FeatureUtils::ComputeCrossMatches(top_scale_desc1, top_scale_desc2, matches, distance_ratio_);
        }
        else
        {
            FeatureUtils::ComputeMatches(top_scale_desc1, top_scale_desc2, matches, distance_ratio_);
        }

        if(matches.size() >= preemtive_min_num_matches_)
            filtered_image_pairs.push_back(image_pair);

    }

    return filtered_image_pairs;

}
cv::Mat BruteFeatureMatcher::GetTopScaleDescriptors(const image_t &image_id)
{
    if(HasTopScaleDescriptorsCache(image_id))
    {
        return top_scale_descriptors_cache_[image_id];
    }
    else
    {
        std::vector<cv::KeyPoint> kpts = database_->ReadKeyPoints(image_id);
        cv::Mat descriptors = database_->ReadDescriptors(image_id);

        cv::Mat top_scale_descriptors;
        FeatureUtils::ExtractTopScaleDescriptors(kpts, descriptors, preemtive_num_features_, top_scale_descriptors);

        top_scale_descriptors_cache_[image_id] = top_scale_descriptors;

        return top_scale_descriptors;
    }
}

bool BruteFeatureMatcher::HasTopScaleDescriptorsCache(const image_t& image_id)
{
    return top_scale_descriptors_cache_.count(image_id) > 0;
}
