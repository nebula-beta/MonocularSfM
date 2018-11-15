






set(MonocularSfM_LIBS
    reconstruction
    bundleadjustment
    visualization
    feature
    common
    database
    sqlite3
    -lpthread
    dl
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES})




add_executable(ImageMatchTest ImageMatchTest.cpp)
target_link_libraries(ImageMatchTest  common feature ${OpenCV_LIBS})


add_executable(FeatureExtraction FeatureExtraction.cpp)
target_link_libraries(FeatureExtraction  feature common   database sqlite3 -lpthread dl ${OpenCV_LIBS})


# feature common 才行
# common feature 不行,  为什么???
# 可能是因为依赖关系吧, 前面的依赖后面的
add_executable(ComputeSequentialMatches ComputeSequentialMatches.cpp)
target_link_libraries(ComputeSequentialMatches  feature common   database sqlite3 -lpthread dl ${OpenCV_LIBS})


add_executable(ComputeBruteMatches ComputeBruteMatches.cpp)
target_link_libraries(ComputeBruteMatches  ${MonocularSfM_LIBS})


add_executable(CheckMatches CheckMatches.cpp)
target_link_libraries(CheckMatches  ${MonocularSfM_LIBS})


add_executable(Reconstruction Reconstruction.cpp)
target_link_libraries(Reconstruction ${MonocularSfM_LIBS} )
