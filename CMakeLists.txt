cmake_minimum_required(VERSION 2.8.8)

find_package(Caffe)
find_package( OpenCV REQUIRED )
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_executable(merger merger.cpp)
target_link_libraries(merger ${Caffe_LIBRARIES}  ${OpenCV_LIBS})
