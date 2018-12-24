# MonocularSfM : Monocular Structure from Motion

## Introuction

MonocularSfm是一个三维重建的程序, 可以对有序或者无序的图片进行三维重建.

程序的输入是**图片**和**相机参数**(包括相机内参`fx`, `fy`, `cx`, `fy`和畸变参数`k1`, `k2`, `p1`, `p2`[可选]).

程序的输出是**三维稀疏点云**和已注册图像的**投影矩阵**.


### south-building
![Image text](./docs/result1.png)


### person-hall
![Image text](./docs/result2.png)

### 东北大学
![Image text](./docs/result4.png)
![Image text](./docs/result5.png)

Number points3D			: `542084`

Number images			: `1329`

Mean reprojection error : `0.33772 [px]`


## Dependencies
* [Eigen](http://eigen.tuxfamily.org) version 3.2
* [OpenCV](http://opencv.org) version 3.x or higher
* [Ceres](http://ceres-solver.org) version 1.10 or higher

## Building
```
mkdir build && cd build
cmake ..
make -j3
```

## How to Run
```
# step1 : 提取特征
./FeatureExtraction image_path database_path  
# ./FeatureExtraction ./images ./database.db
# 表示读取images文件夹中的所有图片,提取特征, 存储到当前目录的database.db

# step2 : 计算匹配(可以使用**顺序匹配**或者是**暴力匹配**)
./ComputeSequentialMatches database_path   # 顺序匹配
./ComputeBruteMatches      database_path   # 暴力匹配

# step3 : 检查匹配, 通过显示不同图像之间的匹配对, 来确认前两步是否正确(可跳过).
./CheckMatches  database_path

# step4 : 重建
./Reconstruction database_path fx fy cx cy output_path
or
./Reconstruction database_path fx fy cx cy k1 k2 p1 p2 output_path

# ./Reconstruction ./database.db fx fy cx cy ./
# 表示从数据库中读取特征及其匹配,然后开始重建, 将重建的结果存储在当前文件夹下"./"
```

## Knowledge
 See the [wiki](https://github.com/nebula-beta/MonocularSfM/wiki) page

## Citations
[1] Snavely N, Seitz S M, Szeliski R. [Photo Tourism: Exploring Photo Collections In 3D](http://phototour.cs.washington.edu/Photo_Tourism.pdf)[J]. Acm Transactions on Graphics, 2006, 25(3):págs. 835-846.

[2] Wu C. [Towards Linear-Time Incremental Structure from Motion](http://ccwu.me/vsfm/vsfm.pdf)[C]// International Conference on 3d Vision. IEEE Computer Society, 2013:127-134.

[3] Schönberger J L, Frahm J M. [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf)[C]// Computer Vision and Pattern Recognition. IEEE, 2016.
