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

#define IMPORT_CONFIDENCE 0


#ifdef COMPILE_ZED
	//ZED Includes
	#include <zed/Camera.hpp>
	#include <zed/utils/GlobalDefine.hpp>
	using namespace sl::zed;
	#define MAX_DEPTH_ZED 10000
#endif

using namespace caffe;  
using std::string;
using boost::shared_ptr;
using namespace boost::filesystem;


cv::Point_<float> center_weight;
cv::Mat weight_mat;
std::vector<cv::Point_<int>> coordinates;	

std::string path_to_protofile  ;
std::string path_to_modelcaffe ;
std::string path_to_solverfile ;
int user_input;
std::vector<cv::Mat> input_channels;
std::string path_image;
int nchannels;
cv::Mat image, image_f;
cv::Mat depth_cnn_rescaled, depth_zed_rescaled, depth_err_rescaled;
float cum_err = 0.0;
bool zed_input = false;
bool vid_input = false;
bool im_seq    = false;
std::string path_video;
std::string images_path_format;
std::string path_frame;
std::string path_depth;
cv::VideoCapture vid;
std::clock_t start;
double duration;
bool save_image;
int frame_counter = 0;
int display = 1;
std::string act_window = "Monocular estimation";
Err_func errors;
int val_points = 0;
int width;
int height;
bool quit = false;

//Variables caffe/CNN	
bool lstm = false;
bool eigen = false;
bool fcn = false;
cv::Size input_geometry_;
shared_ptr<Net<float> > net_;
FILE *file_proto, *file_caffe_model,*file_train_proto, *file_solver_proto ,*file_image;
Blob<float>* input_layer;
Blob<float>* output_layer;
caffe::MemoryDataLayer<float> *dataLayer_trainnet;
caffe::MemoryDataLayer<float> *dataLayer_testnet;
shared_ptr<caffe::Blob<float>> output_norm; 
const float* begin_mem_output;
double min_depth_cnn,  max_depth_cnn;
float* input_data;
cv::Size output_geo;
 cv::Size output_geo_eigen;
cv::Mat depth_cnn, depth_cnn_un; 

//Variables ZED	4
#ifdef COMPILE_ZED

SENSING_MODE dm_type = FULL;
DATA_TYPE dtype = FLOAT;
MAT_TYPE mtype = CPU;
sl::zed::Camera* zed;
double min_depth_zed, max_depth_zed;
double min_depth_err, max_depth_err;
double min_conf, max_conf;
float fx;
float baseline;

#endif