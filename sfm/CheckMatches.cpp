#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "Database/Database.h"
#include "Feature/FeatureUtils.h"

using namespace std;
using namespace cv;
using namespace MonocularSfM;


std::string UnionPath(const std::string& directory, const std::string& filename)
{
    if(directory[directory.size() - 1] == '/')
    {
        return directory + filename;
    }
    else
    {
        return directory + "/" + filename;
    }
}
int main(int argc, char** argv)
{


    // step3 : 从数据库读取匹配, 并显示在图片上, 检查匹配是否正确


    if(argc != 2)
    {
        std::cout << "You need specify the YAML file path!" << std::endl;
        exit(-1);
    }


    cv::FileStorage fs(argv[1], FileStorage::READ);

    if(!fs.isOpened())
    {
        std::cout << "YAML file : " << argv[1] << " can't not open!" << std::endl;
        exit(-1);
    }

    string images_path;
    string database_path;

    fs["images_path"] >> images_path;
    fs["database_path"] >> database_path;



    Database database ;
    database.Open(database_path);


    std::vector<std::pair<image_pair_t, std::vector<cv::DMatch>>> all_matches = database.ReadAllMatches();

    for(const auto& matches : all_matches)
    {
        image_t image_id1, image_id2;
        Database::PairIdToImagePair(matches.first, &image_id1, &image_id2);

        Database::Image image1 = database.ReadImageById(image_id1);
        Database::Image image2 = database.ReadImageById(image_id2);

        cv::Mat cv_image1 = cv::imread(UnionPath(images_path, image1.name));
        cv::Mat cv_image2 = cv::imread(UnionPath(images_path, image2.name));

        std::vector<cv::KeyPoint> kpts1 = database.ReadKeyPoints(image_id1);
        std::vector<cv::KeyPoint> kpts2 = database.ReadKeyPoints(image_id2);

        std::vector<cv::Point2f> pts1, pts2;
        cv::KeyPoint::convert(kpts1, pts1);
        cv::KeyPoint::convert(kpts2, pts2);
        std::cout << image_id1 << " -- " << image_id2 << " : " << matches.second.size() << std::endl;
        FeatureUtils::ShowMatches(cv_image1, cv_image2, pts1, pts2, matches.second, "name", 1);

    }


    database.Close();
    return 0;
}
