//My includes
#include "my_func.h"

//Caffe includes
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
#include <cstdio>

//ZED Includes
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

using namespace caffe;  
using std::string;
using boost::shared_ptr;
using namespace boost::filesystem;
using namespace sl::zed;

