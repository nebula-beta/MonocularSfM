#include <iostream>

#include <opencv2/opencv.hpp>

#include "Database/Database.h"
#include "Feature/FeatureUtils.h"

using namespace std;
using namespace cv;
using namespace MonocularSfM;

int main()
{


    // step3 : 从数据库读取匹配, 并显示在图片上, 检查匹配是否正确



    // TODO 将path变成命令行参数的形式

    Database database ;
    database.Open("./person-hall2.db");


    std::vector<std::pair<image_pair_t, std::vector<cv::DMatch>>> all_matches = database.ReadAllMatches();

    for(const auto& matches : all_matches)
    {
        image_t image_id1, image_id2;
        Database::PairIdToImagePair(matches.first, &image_id1, &image_id2);

        Database::Image image1 = database.ReadImageById(image_id1);
        Database::Image image2 = database.ReadImageById(image_id2);

        cv::Mat cv_image1 = cv::imread(image1.name);
        cv::Mat cv_image2 = cv::imread(image2.name);

        std::vector<cv::KeyPoint> kpts1 = database.ReadKeyPoints(image_id1);
        std::vector<cv::KeyPoint> kpts2 = database.ReadKeyPoints(image_id2);

        std::vector<cv::Point2f> pts1, pts2;
        cv::KeyPoint::convert(kpts1, pts1);
        cv::KeyPoint::convert(kpts2, pts2);
        std::cout << image_id1 << " -- " << image_id2 << " : " << matches.second.size() << std::endl;
        FeatureUtils::ShowMatches(image1.name, image2.name, pts1, pts2, matches.second, "name", 1);


    }


    database.Close();
    return 0;
}
