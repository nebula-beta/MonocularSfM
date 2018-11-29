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

    assert(argc == 3);



    // 图片所在的文件夹
    string images_path = argv[1];
    string database_path = argv[2];


    int max_image_size = 3200;
    int num_features = 8024;


    Timer timer;
    timer.Start();

    cv::Ptr<FeatureExtractor> extractor =
            cv::Ptr<FeatureExtractorCPU>(new FeatureExtractorCPU(database_path, images_path, max_image_size, num_features));
    extractor->RunExtraction();

    timer.PrintMinutes();



    return 0;
}
