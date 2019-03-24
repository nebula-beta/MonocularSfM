set(MonocularSfM_Feature_LIBS

    feature
    common
    database
    sqlite3
    -lpthread
    dl
    ${OpenCV_LIBS}
)


set(MonocularSfM_Reconstruction_LIBS
    reconstruction
    optimizer
    visualization
    common
    database
    sqlite3
    -lpthread
    dl
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)

add_executable(FeatureExtraction FeatureExtraction.cpp)
target_link_libraries(FeatureExtraction ${MonocularSfM_Feature_LIBS})

add_executable(ComputeMatches ComputeMatches.cpp)
target_link_libraries(ComputeMatches ${MonocularSfM_Feature_LIBS})


add_executable(CheckMatches CheckMatches.cpp)
target_link_libraries(CheckMatches ${MonocularSfM_Feature_LIBS})


add_executable(Reconstruction Reconstruction.cpp)
target_link_libraries(Reconstruction ${MonocularSfM_Reconstruction_LIBS})



