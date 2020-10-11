#include <unordered_map>

#include <opencv2/xfeatures2d.hpp>

#include "Feature/FeatureUtils.h"


using namespace MonocularSfM;





void FeatureUtils::ExtractFeature(const cv::Mat& image,
                                   std::vector<cv::KeyPoint>& kpts,
                                   cv::Mat& desc,
                                   int num_features)
{
    /// note :      不能使用cv::xfeatures2d::SIFT::create(num_features)来限制特征点的数量
    ///             当图片的分辨率小时，出错
    ///             但是当图片的分辨率大时，　会导致匹配的数量急剧下降
    /// 初步估计     opencv可能是根据响应值分数从大到小对特征点进行排序，　然后保留分数大（稳定的)特征点
    ///             也许不同的图片取得的排序靠前的特征点，　并不是重复的．
    ///             也是说大部分的特征点在这张图片上排序靠前，　在其它图片上排序靠后

    //cv::Ptr<cv::xfeatures2d::SIFT> ptr = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::SIFT> ptr = cv::SIFT::create();
    std::vector<cv::KeyPoint> tmp_kpts;
    std::vector<cv::KeyPoint> top_scale_kpts;
    ptr->detect(image, tmp_kpts);
    FeatureUtils::ExtractTopScaleKeyPoints(tmp_kpts, num_features, top_scale_kpts);
    ptr->compute(image, top_scale_kpts, desc);
    kpts = top_scale_kpts;

//    cv::KeyPoint::convert(top_scale_kpts, pts);
}

void FeatureUtils::ExtractTopScaleKeyPoints(const std::vector<cv::KeyPoint> kpts,
                                            const int& num_features,
                                            std::vector<cv::KeyPoint>& top_scale_kpts)
{
    if(num_features > kpts.size())
    {
        top_scale_kpts = kpts;
    }
    else
    {
        std::vector<std::pair<size_t, float>> scales;
        for(size_t i = 0; i < kpts.size(); ++i)
        {
            scales.emplace_back(i, kpts[i].size);
        }
        std::partial_sort(scales.begin(), scales.begin() + num_features,
                          scales.end(), [](const std::pair<size_t, float> scale1,
                                           const std::pair<size_t, float> scale2)
                                            {return scale1.second > scale2.second;}
                            );

        top_scale_kpts.reserve(num_features);
        for(size_t i = 0; i < num_features; ++i)
        {
            top_scale_kpts.push_back(kpts[scales[i].first]);
        }
    }
}


void FeatureUtils::ExtractTopScaleDescriptors(const std::vector<cv::KeyPoint> kpts,
                                              const cv::Mat& descriptors,
                                              const int& num_features,
                                              cv::Mat& top_scale_descriptors)
{
    if(num_features > kpts.size())
    {
        top_scale_descriptors = descriptors;
    }
    else
    {
        std::vector<std::pair<size_t, float>> scales;
        for(size_t i = 0; i < kpts.size(); ++i)
        {
            scales.emplace_back(i, kpts[i].size);
        }
        std::partial_sort(scales.begin(), scales.begin() + num_features,
                          scales.end(), [](const std::pair<size_t, float> scale1,
                                           const std::pair<size_t, float> scale2)
                                            {return scale1.second > scale2.second;}
                            );

        top_scale_descriptors = cv::Mat(num_features, 128, CV_32F);
        for(size_t i = 0; i < num_features; ++i)
        {
            descriptors.row(scales[i].first).copyTo(top_scale_descriptors.row(i));
        }
    }
}


void FeatureUtils::UndistortFeature(const cv::Mat& image,
                                     const cv::Mat& K,
                                     const cv::Mat& dist_coef,
                                     const std::vector<cv::Point2f>& pts,
                                     std::vector<cv::Point2f>& undistort_pts,
                                     std::vector<size_t>& index)
{
    cv::Mat mat(pts.size(), 2, CV_32F);
    for(size_t i = 0; i < pts.size(); ++i)
    {
        mat.at<float>(i, 0) = pts[i].x;
        mat.at<float>(i, 1) = pts[i].y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K, dist_coef, cv::Mat(), K);
    mat = mat.reshape(1);

    undistort_pts.clear();
    index.clear();
    for(size_t i = 0; i < pts.size(); ++i)
    {
        cv::Point2f pt(mat.at<float>(i, 0), mat.at<float>(i, 1));


//        if(pt.x < 0 || pt.y < 0 || pt.x > image.cols - 5 || pt.y > image.rows - 5)
//            continue;

//        std::cout << "(" << pts[i].x << ", " << pts[i].y << ") -----> (" << pt.x << ", " << pt.y << ")" << std::endl;

        undistort_pts.push_back(pt);
        index.push_back(i);
    }
}








void FeatureUtils::ComputeMatches(const cv::Mat& desc1,
                                   const cv::Mat& desc2,
                                   std::vector<cv::DMatch>& matches,
                                   const float distance_ratio)
{
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
    std::vector<std::vector<cv::DMatch>> initial_matches;

    matcher->knnMatch(desc1, desc2, initial_matches, 2);
    for(auto& m : initial_matches)
    {
        if(m[0].distance < distance_ratio * m[1].distance)
        {
            matches.push_back(m[0]);
        }
    }
}


void FeatureUtils::ComputeCrossMatches(const cv::Mat& desc1,
                                        const cv::Mat& desc2,
                                        std::vector<cv::DMatch>& matches,
                                        const float distance_ratio)
{
    std::vector<cv::DMatch> matches12;
    std::vector<cv::DMatch> matches21;

    ComputeMatches(desc1, desc2, matches12, distance_ratio);
    ComputeMatches(desc2, desc1, matches21, distance_ratio);


    CrossCheck(matches12, matches21, matches);

}

void FeatureUtils::FilterMatches(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  const std::vector<cv::DMatch>& matches,
                                  std::vector<cv::DMatch>& prune_matches)
{
    if(pts1.size() == 0 || matches.size() == 0)
    {
        return;
    }


    std::vector<cv::Point2f> aligned_pts1, aligned_pts2;
    cv::Mat inlier_mask;
    GetAlignedPointsFromMatches(pts1, pts2, matches, aligned_pts1, aligned_pts2);

    if(aligned_pts1.size() == 0)
    {
        return;
    }

    cv::findFundamentalMat(aligned_pts1, aligned_pts2, cv::FM_RANSAC, 3.0, 0.99, inlier_mask);

    int good_matches = 0;
    for(int i = 0; i < inlier_mask.rows; ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0) continue;
        good_matches += 1;
        prune_matches.push_back(matches[i]);
    }
//    std::cout << "after filter, matches = [" << good_matches << " / " << matches.size() << "]" << std::endl;
}

void FeatureUtils::FilterMatchesByDistance(const std::vector<cv::DMatch>& matches,
                                           std::vector<cv::DMatch>& prune_matches,
                                           const double& max_distance)
{
    for(size_t i = 0; i < matches.size(); ++i)
    {
        if(matches[i].distance > max_distance)
            continue;
        prune_matches.push_back(matches[i]);
    }
}


void FeatureUtils::ShowMatches(const cv::Mat& image1,
                                const cv::Mat& image2,
                                const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const std::vector<cv::DMatch>& matches,
                                const std::string& window_name,
                                const time_t duration)
{
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::Mat canvas = image1.clone();
    canvas.push_back(image2.clone());
    cv::RNG rng;

    for(size_t i = 0; i < matches.size(); ++i)
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx;
        int r = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int b = rng.uniform(0, 255);
        cv::line(canvas, pts1[queryIdx], pts2[trainIdx]+ cv::Point2f(0, image1.rows), cv::Scalar(r, g, b), 2);
    }

    cv::imshow(window_name, canvas);
    cv::waitKey(duration);
}

void FeatureUtils::ShowMatches(const std::string& image_name1,
                                const std::string& image_name2,
                                const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& pts2,
                                const std::vector<cv::DMatch>& matches,
                                const std::string& window_name,
                                const time_t duration)
{
    cv::Mat image1 = cv::imread(image_name1);
    cv::Mat image2 = cv::imread(image_name2);

    FeatureUtils::ShowMatches(image1, image2, pts1, pts2, matches, window_name, duration);
}

void FeatureUtils::GetAlignedPointsFromMatches(const std::vector<cv::Point2f>& pts1,
                                                const std::vector<cv::Point2f>& pts2,
                                                const std::vector<cv::DMatch>& matches,
                                                std::vector<cv::Point2f>& aligned_pts1,
                                                std::vector<cv::Point2f>& aligned_pts2)
{
    aligned_pts1.reserve(matches.size());
    aligned_pts2.reserve(matches.size());

    for(size_t i = 0; i < matches.size(); ++i)
    {
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx;

        aligned_pts1.push_back(pts1[queryIdx]);
        aligned_pts2.push_back(pts2[trainIdx]);
    }
}

void FeatureUtils::CrossCheck(const  std::vector<cv::DMatch>& matches12,
                               const std::vector<cv::DMatch>& matches21,
                               std::vector<cv::DMatch>& prune_matches)
{
    std::unordered_map<int, int> vis;


    for(size_t i = 0; i < matches21.size(); ++i)
    {
        int query_idx = matches21[i].queryIdx;
        int train_idx = matches21[i].trainIdx;
        vis[query_idx] = train_idx;
    }

    int good_matches = 0;
    for(size_t i = 0; i < matches12.size(); ++i)
    {

       int query_idx = matches12[i].queryIdx;
       int train_idx = matches12[i].trainIdx;

       if(vis[train_idx] == query_idx)
       {
           good_matches += 1;
           prune_matches.push_back(matches12[i]);
       }

    }
//    std::cout << "after cross check, matches = [" << good_matches << " / " << matches12.size() << "]" << std::endl;
}



