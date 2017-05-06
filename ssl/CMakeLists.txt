cmake_minimum_required(VERSION 2.8.8)

find_package(Caffe REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZED 0.9)
find_package(CUDA )

IF(ZED_FOUND AND CUDA_FOUND)
include_directories(${ZED_INCLUDE_DIRS})
link_directories(${ZED_LIBRARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
add_definitions(-DCOMPILE_ZED)
add_definitions(-DCOMPILE_CUDA)
ENDIF()

include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/includes)

add_definitions(${Caffe_DEFINITIONS})  


add_executable(ssl "../src/ssl.cpp" "../src/includes/manageObjectInputMap.cpp" "../src/includes/manageObjectCnn.cpp" "../src/includes/manageObjectDepthMap.cpp" "../src/includes/facilMergeAlgorithm.cpp" "../src/includes/manageDepthMapPerformance.cpp")
add_definitions(-std=c++0x)


IF(ZED_FOUND AND CUDA_FOUND)
target_link_libraries(ssl ${Caffe_LIBRARIES}  ${OpenCV_LIBS} ${ZED_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})
ELSE()
target_link_libraries(ssl ${Caffe_LIBRARIES}  ${OpenCV_LIBS})
ENDIF()
