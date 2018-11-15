#include <iostream>
#include <opencv2/opencv.hpp>

#include "Common/Timer.h"
#include "Reconstruction/Mapper.h"
using namespace std;
using namespace cv;
using namespace  MonocularSfM;


int main()
{

    // step4 : 重建



    // TODO 将path变成命令行参数的形式




    Timer timer;
    timer.Start();

    Mapper::Config config;

    // person_hall
//    string database_path = "./database.db";
//    string database_path = "./person-hall2.db";

//    config.fx = 3839.71;
//    config.fy = 3840.23;
//    config.cx = 2808;
//    config.cy = 1872;

//    config.k1 = -0.109344;
//    config.k2 = 0.0790394;
//    config.p1 = 0.000101365;
//    config.p2 = 0.000233581;

    // templeRing
//    string database_path = "./templeRing.db";
//    config.fx = 3838.27;
//    config.fy = 3837.22;
//    config.cx = 2808;
//    config.cy = 1872;

//    config.k1 = -0.110339;
//    config.k2 = 0.079547;
//    config.p1 = 0.000116211;
//    config.p2 = 0.00029483;


     // south-building
    string database_path = "./south-building.db";
    config.fx = 2559.68;
    config.fy = 2559.68;
    config.cx = 1536;
    config.cy = 1152;

    config.k1 = -0.0204997;
    config.k2 = 0;
    config.p1 = 0;
    config.p2 = 0;

    config.min_num_matches = 15;

    Mapper mapper(database_path, config);
    mapper.DoMapper();




    timer.PrintMinutes();

    return 0;
}

