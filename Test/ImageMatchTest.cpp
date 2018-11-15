#include <iostream>
#include <string>
#include "Feature/FeatureUtils.h"

using namespace std;
using namespace cv;
using namespace MonocularSfM;

void scaleImage(cv::Mat& src, cv::Mat& dst, const int& max_image_size, double& scale_x, double& scale_y)
{
    if(max_image_size < src.rows || max_image_size < src.cols)
    {
        const int width = src.cols;
        const int height = src.rows;
        const double scale = max_image_size * 1.0 / std::max(width, height);
        const int new_width = width * scale;
        const int new_height = height * scale;

        scale_x = new_width * 1.0 / width;
        scale_y = new_height * 1.0 / height;

        cv::resize(src, dst, cv::Size(new_width, new_height));
    }
    else
    {
        scale_x = 1.0;
        scale_y = 1.0;
        dst = src.clone();
    }
}

int main()
{
    string image_path1 = "/home/anton/Desktop/templeRing/IMG_2331.JPG";
    string image_path2 = "/home/anton/Desktop/templeRing/IMG_2332.JPG";

    cv::Mat image1 = imread(image_path1);
    cv::Mat image2 = imread(image_path2);
    cv::Mat scaled_image1;
    cv::Mat scaled_image2;
    double scale_x;
    double scale_y;
    const int max_image_size = 3200;

    scaleImage(image1, scaled_image1, max_image_size, scale_x, scale_y);
    scaleImage(image2, scaled_image2, max_image_size, scale_x, scale_y);



    vector<cv::KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    // 如果限制了特征点的数量，　那么会导致接下来的匹配变得很少
    // 也就是说不同的图片取得的排序靠前的特征点，　并不是重复的．
    // 也是说大部分的特征点在这张图片上排序靠前，　在其它图片上排序靠后
    // 但是如果图片的尺寸不是很大的话，　那么就没有这个问题
    FeatureUtils::ExtractFeature(scaled_image1, kpts1, desc1, 10240);
    FeatureUtils::ExtractFeature(scaled_image2, kpts2, desc2, 10240);

    std::cout << "num1 " << kpts1.size() << std::endl;

    std::cout << "num2 " << kpts2.size() << std::endl;

    vector<cv::DMatch> matches;
    FeatureUtils::ComputeCrossMatches(desc1, desc2, matches, 0.8);

    std::cout << "Matches num " << matches.size() << std::endl;

    vector<cv::Point2f> pts1, pts2;
    cv::KeyPoint::convert(kpts1, pts1);
    cv::KeyPoint::convert(kpts2, pts2);

    FeatureUtils::ShowMatches(scaled_image1, scaled_image2, pts1, pts2, matches, "name", 0);
    return 0;
}
