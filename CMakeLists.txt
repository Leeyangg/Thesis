cmake_minimum_required(VERSION 2.8.8)

find_package(Caffe REQUIRED)
find_package( OpenCV REQUIRED )
find_package(ZED 0.9 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

add_definitions(${Caffe_DEFINITIONS})  

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)

#cuda_add_executable(merger ${SRC_FILES} ${SRC_CU_FOLDER})

add_executable(merger "../src/merger.cpp" "../src/include/my_func.cpp")
add_definitions(-std=c++0x) # -m64)
target_link_libraries(merger ${Caffe_LIBRARIES}  ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})
