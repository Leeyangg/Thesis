#include <caffe/caffe.hpp>
#include "caffe/layers/memory_data_layer.hpp"
#include <caffe/sgd_solvers.hpp>

//opencv includes
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//standard includes
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <math.h>

//ZED Includes
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

#define NORMALIZATION_FACTOR 10.0
#define ZED_NORMALIZATION_FACTOR 39.75

cv::Mat merge(std::vector<cv::Point_<int>> coord, cv::Mat stereo, cv::Mat mono,cv::Mat weight_mat , cv::Point_<float>* center_weight );

void plot_maps(cv::Mat map, float scale_factor, cv::Size geometry, int color_map, const char* name_window);


