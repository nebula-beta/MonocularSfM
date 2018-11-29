#include <iostream>
#include <opencv2/opencv.hpp>

#include "Common/Timer.h"
#include "Database/Database.h"
#include "Feature/FeatureMatching.h"
using namespace std;
using namespace cv;
using namespace  MonocularSfM;


int main(int argc, char** argv)
{

    // step2 : 图片之间进行暴力匹配, 并存储到数据库

    assert(argc == 2);

    string database_path = argv[1];

    Timer timer;
    timer.Start();

    BruteFeatureMatcher brute_matcher(database_path);
    // 进行序列式匹配
    brute_matcher.RunMatching();

    timer.PrintMinutes();

    return 0;
}
