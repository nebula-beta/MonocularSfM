#include <iostream>
#include <opencv2/opencv.hpp>

#include "Common/Timer.h"
#include "Database/Database.h"
#include "Feature/FeatureExtraction.h"
using namespace std;
using namespace cv;
using namespace MonocularSfM;

int main(int argc, int** argv)
{
    // step1 : 提取特征, 并存储到数据库




    // TODO 将参数变成命令行参数的形式

    // 图片所在的文件夹
    string images_path = "/home/anton/workspace/resources/clomap/person-hall/images/";
    string database_path = "./person-hall2.db";



    int max_image_size = 3200;
    int num_features = 20240;


    Timer timer;
    timer.Start();
    // TODO : 何时用指针, 何时用智能指针, 何时不用指针???
    FeatureExtractor* extractor = new FeatureExtractorCPU(database_path, images_path, max_image_size, num_features);
    extractor->RunExtraction();

    timer.PrintMinutes();

    delete extractor;


    return 0;
}


