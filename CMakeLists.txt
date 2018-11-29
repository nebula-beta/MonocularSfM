cmake_minimum_required(VERSION 2.8)

project(FinalMonocularSfM)



# 用IF控制，可以实现Release版本不输出debug信息
#不知道为什么DEBUG必须写成-DDEBUG
IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()


#cmake  -DCMAKE_BUILD_TYPE=Debug ..


# 设置可执行文件在那个目录下生成
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 头文件的搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})

# 设置lib库在那个目录下生成
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
link_directories(${PROJECT_SOURCE_DIR/lib})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")



find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


add_subdirectory(src)
add_subdirectory(ext)
add_subdirectory(UnitTest)
add_subdirectory(sfm)


