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
add_executable(TimerTest TimerTest.cpp)
target_link_libraries(TimerTest  ${MonocularSfM_Feature_LIBS})


add_executable(InitializerTest InitializerTest.cpp)
target_link_libraries(InitializerTest   ${MonocularSfM_Reconstruction_LIBS})


add_executable(RegisterGraphTest RegisterGraphTest.cpp)
target_link_libraries(RegisterGraphTest   ${MonocularSfM_Reconstruction_LIBS})


add_executable(MapBuilderTest MapBuilderTest.cpp)
target_link_libraries(MapBuilderTest   ${MonocularSfM_Reconstruction_LIBS})


