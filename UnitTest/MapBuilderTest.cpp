#include "Reconstruction/MapBuilder.h"
#include <string>
using namespace std;
using namespace MonocularSfM;

int main()
{

    MapBuilder::Parameters params;

//    string database_path = "/media/anton/软件/image_all/image_all.db";

//    params.fx = 3666.666504 / 2.53;
//    params.fy = 3666.666504 / 2.53;
//    params.cx = 1080;
//    params.cy = 720;

//    params.fx = 3666.666504;
//    params.fy = 3666.666504;
//    params.cx = 5472.0 / 2;
//    params.cy = 3078.0 / 2;


//    string database_path = "/Users/anton/gerrard-hall.db";
//    // 相机内参
//    params.fx = 3838.27;
//    params.fy = 3837.22;
//    params.cx = 2808;
//    params.cy = 1872;

//    // 畸变参数
//    params.k1 = -0.110339;
//    params.k2 = 0.079547;
//    params.p1 = 0.000116211;
//    params.p2 = 0.00029483;

    string database_path = "/Users/anton/person-hall-sequential.db";

    // 相机内参
    params.fx = 3839.71;
    params.fy = 3840.23;
    params.cx = 2808;
    params.cy = 1872;

    // 畸变参数
    params.k1 = -0.109344;
    params.k2 = 0.0790394;
    params.p1 = 0.000101365;
    params.p2 = 0.000233581;


    // south-building
//    string database_path = "/home/anton/workspace/resources/clomap/south-building/south-building.db";
//    params.fx = 2559.68;
//    params.fy = 2559.68;
//    params.cx = 1536;
//    params.cy = 1152;

//    params.k1 = -0.0204997;
//    params.k2 = 0;
//    params.p1 = 0;
//    params.p2 = 0;

//    string database_path = "/home/anton/workspace/resources/clomap/graham-hall2/exterior-brute.db";

//    // 相机内参
//    params.fx = 3881;
//    params.fy = 3881;
//    params.cx = 2808;
//    params.cy = 1872;

//    // 畸变参数
//    params.k1 = -0.0638997;
//    params.k2 = 0;
//    params.p1 = 0;
//    params.p2 = 0;

    MapBuilder map_builder(database_path, params);

    map_builder.SetUp();
    map_builder.DoBuild();
    map_builder.WritePLY("./points3D.ply");
    map_builder.WritePLYBinary("./points3D_binary.ply");
    map_builder.Write("./");
}
