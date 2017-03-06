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

#define NORMALIZATION_FACTOR 39.75
#define CNN_NORMALIZATION_FACTOR 39.75
#define THRESHOLD_CONFIDENCE 0.8

cv::Mat merge(std::vector<cv::Point_<int>> coord, cv::Mat stereo, cv::Mat mono,cv::Mat weight_mat , cv::Point_<float>* center_weight );

void plot_maps(cv::Mat map, float scale_factor, cv::Size geometry, int color_map, const char* name_window, bool save_image);

class Err_func{

	public:
		Err_func();
		cv::Mat map_gt;
		cv::Mat map_pred;

		void push_mat_gt(cv::Mat push_mat){push_mat.copyTo(map_gt);}
		void push_mat_pred(cv::Mat push_mat){push_mat.copyTo(map_pred);}
		float get_error(std::string error_func);

	private:	

		float threshold_err(cv::Mat map_gt, cv::Mat map_pred);
		float abs_rel_diff(cv::Mat map_gt, cv::Mat map_pred);
		float sqr_rel_diff(cv::Mat map_gt, cv::Mat map_pred);
		float rmse_lin(cv::Mat map_gt, cv::Mat map_pred);
		float rmse_log(cv::Mat map_gt, cv::Mat map_pred);
		float rmse_log_inv(cv::Mat map_gt, cv::Mat map_pred);

};

