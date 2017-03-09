cmake_minimum_required(VERSION 2.8.8)

find_package(Caffe REQUIRED)
find_package( OpenCV REQUIRED )


include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

add_definitions(${Caffe_DEFINITIONS})  


add_executable(data_set "../src/data_set.cpp")
add_definitions(-std=c++0x) # -m64)
target_link_libraries(data_set ${Caffe_LIBRARIES}  ${OpenCV_LIBS} )
