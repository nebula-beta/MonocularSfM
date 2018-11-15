# MonocularSfM : Monocular Structure from Motion

## Introuction

MonocularSfm是一个三维重建的程序, 可以对有序或者无序的图片进行三维重建.

程序的输入是**图片**和**相机参数**(包括相机内参`fx`, `fy`, `cx`, `fy`和畸变参数`k1`, `k2`, `p1`, `p2`[可选]).

程序的输出是**三维稀疏点云**和已注册图像的**投影矩阵**.

![Image text](./docs/result1.png)


## Dependencies
* [Eigen](http://eigen.tuxfamily.org) version 3.2
* [OpenCV](http://opencv.org) version 3.x or higher
* [Ceres](http://ceres-solver.org) version 1.10 or higher

## Building
```
mkdir build
cmake ..
make -j2
```

## How to Run
```
# step1 : 提取特征
./FeatureExtraction   

# step2 : 计算匹配(可以使用**顺序匹配**或者是**暴力匹配**)
./ComputeSequentialMatches  # 顺序匹配
./ComputeBruteMatches       # 暴力匹配

# step3 : 检查匹配, 通过显示不同图像之间的匹配对, 来确认前两步是否正确(可跳过).
./CheckMatches

# step4 : 重建
./Reconstruction

```

## Citations

