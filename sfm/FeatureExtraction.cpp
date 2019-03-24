#include <iostream>
#include <opencv2/opencv.hpp>

#include "Common/Timer.h"
#include "Database/Database.h"
#include "Feature/FeatureExtraction.h"
using namespace std;
using namespace cv;
using namespace MonocularSfM;

int main(int argc, char** argv)
{
    // step1 : 提取特征, 并存储到数据库

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

    // 图片所在的文件夹
    string images_path;
    string database_path;

    int max_image_size = 3200;
    int num_features = 8024;
    int normalization_type = 0;

    fs["images_path"] >> images_path;
    fs["database_path"] >> database_path;
    fs["SIFTextractor.max_image_size"] >> max_image_size;
    fs["SIFTextractor.num_features"] >> num_features;
    fs["SIFTextractor.normalization"] >> normalization_type;

    assert(max_image_size > 0);
    assert(num_features > 0);
    assert(normalization_type == 0 || normalization_type == 1 || normalization_type == 2);

    Timer timer;
    timer.Start();

    cv::Ptr<FeatureExtractor> extractor;

    if(normalization_type == 0)
    {
        extractor = cv::Ptr<FeatureExtractorCPU>(new FeatureExtractorCPU(database_path, images_path, max_image_size, num_features,
                                                                         FeatureExtractor::Normalization::L1_ROOT));

    }
    else if(normalization_type == 1)
    {
        extractor = cv::Ptr<FeatureExtractorCPU>(new FeatureExtractorCPU(database_path, images_path, max_image_size, num_features,
                                                                         FeatureExtractor::Normalization::L2));
    }
    else
    {
        extractor = cv::Ptr<FeatureExtractorCPU>(new FeatureExtractorCPU(database_path, images_path, max_image_size, num_features,
                                                                         FeatureExtractor::Normalization::ROOT_SIFT));
    }
    extractor->RunExtraction();

    timer.PrintMinutes();



    return 0;
}
