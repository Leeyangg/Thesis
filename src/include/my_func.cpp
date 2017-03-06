#include "my_func.h"

#define THRESHOLD_ERR 1.25
#define CONVERT_SCALE 20000.0/(255*1000.0)

cv::Mat merge(std::vector<cv::Point_<int>> coord, cv::Mat stereo, cv::Mat mono,cv::Mat weight_mat , cv::Point_<float>* center_weight ){

	cv::Mat merged(stereo.rows, stereo.cols,CV_32FC1);
	cv::Mat dx;
   cv::Mat dy;
	std::vector<float>  wnn;
   float w1=0.0, w2=0.0, w3=0.0, w4=0.0, min_wnn =0.0, sum_w=0.0, inter_depth = 0.0, w_norm=0.0;
	float sig1 = 15.0;
	float sig2 = 0.1;
	float sig3 = 1*exp(-3);
   bool first = true;
	float mean_merged_depth = 0.0;
	float cnn_merged_depth = 0.0;
	float zed_merged_depth = 0.0;
   cv::Mat mono_scaled;
 	int aux = 0;

   cv::Sobel(mono,dx, -1, 1, 0, -1, 1, 0, cv::BORDER_DEFAULT);
   cv::Sobel(mono,dy, -1, 0, 1, -1, 1, 0, cv::BORDER_DEFAULT); 

   center_weight->x = coord[0].x ;
	center_weight->y = coord[0].y ;

	for(int h=0 ; h <mono.rows ; h++){

	  for(int w = 0 ; w < mono.cols; w++){

		  //weight_mat.at<float>(h,w) = 0.0;
	     for(int i = 0; i < coord.size(); i++){

		     w1 = exp( (-1*sqrt(  pow((h - coord[i].y),2) +  pow((w - coord[i].x),2) ))/sig1 );
			  w2 = (1/(abs(dx.at<float>(coord[i].y, coord[i].x) - dx.at<float>(h,w)) + sig2)) * (1/(abs(   dy.at<float>(coord[i].y, coord[i].x) - dy.at<float>(h,w) ) + sig2)) ;
           w3 = exp( -abs( mono.at<float>(h,w)  + dx.at<float>(h,w)*(h - coord[i].y) - mono.at<float>(coord[i].y, coord[i].x)  ) ) + sig3;
           w4 = exp( -abs( mono.at<float>(h,w) + dy.at<float>(h,w)*(h - coord[i].y) - mono.at<float>(coord[i].y, coord[i].x)  ) ) + sig3;
           
		     wnn.push_back(w1*w2*w3*w4);
           sum_w = sum_w + w1*w2*w3*w4;
		 	
			  if(first){
			     min_wnn = w1*w2*w3*w4;
              first = false;
         }
	      	else{
					if(w1*w2*w3*w4 < min_wnn)
		   		min_wnn = w1*w2*w3*w4; 
			   }
		  }

	     for(int i = 0; i < coord.size(); i++){
   
		      w_norm = (wnn[i] - min_wnn)/(sum_w - min_wnn);

				inter_depth = inter_depth +  w_norm*( stereo.at<float>(coord[i].y,coord[i].x) + mono.at<float>(h,w) - mono.at<float>(coord[i].y, coord[i].x));

            if(i == 0)
				 	  weight_mat.at<float>(h,w) =  w_norm;


        }

	   merged.at<float>(h,w) =  inter_depth/NORMALIZATION_FACTOR;

		if(inter_depth < 0.0){
		   merged.at<float>(h,w) = -888.0;
      }
	
	   else{
        mean_merged_depth = mean_merged_depth +  inter_depth;
	     cnn_merged_depth =  cnn_merged_depth  +  mono.at<float>(h,w) ;
	     zed_merged_depth =  zed_merged_depth  +  stereo.at<float>(h,w);
	     aux++;
      }
        wnn.clear();
		  w1 = 0.0;
		  w2 = 0.0;
		  w3 = 0.0;
		  w4 = 0.0;
        min_wnn = 0.0;
		  sum_w = 0.0;	
        w_norm = 0.0;
        inter_depth = 0.0;
		  first = true;	
	  }


	}

	   mean_merged_depth = mean_merged_depth/(aux);
	   cnn_merged_depth  = cnn_merged_depth  /(aux);
	   zed_merged_depth  =  zed_merged_depth  /(aux);

      std::cout << (float) aux/(stereo.rows*stereo.cols) << " points used" << "  CNN Merged = " << cnn_merged_depth << "  ZED Merged = " <<  zed_merged_depth << " Merged = " << mean_merged_depth <<  std::endl;

      return merged;
}


void plot_maps(cv::Mat map, float scale_factor, cv::Size geometry, int color_map, const char* name_window, bool save_image){

	cv::Mat map_color(geometry.height, geometry.width, CV_32FC3);
   cv::Mat map_color_bad(geometry.height, geometry.width, CV_32FC3,0.0);
	std::string name( name_window);
   cv::convertScaleAbs(map, map, scale_factor);
	
	if(color_map != -1){

		applyColorMap(map, map_color, color_map);

  /*if(name_window[0] == 'M' && name_window[17] == '2'){
	for(int h=0 ; h <map.rows ; h++){
	  for(int w = 0 ; w < map.cols; w++){
			if(map.at<float>(h,w) < 0.0){
				map_color_bad.at<float>(h,w,0) = 255.0;
				map_color_bad.at<float>(h,w,1) = 255.0;
				map_color_bad.at<float>(h,w,2) = 255.0;
			}
	  }
   }

  }*/

	cv::imshow(name_window, map_color);
  // cv::imshow("Bad pixels", map_color_bad);

	if(save_image)
		cv::imwrite("../images/" +name+ ".jpeg", map_color);
   }

   else{
		cv::imshow(name_window, map);
	if(save_image)
		cv::imwrite("../images/" +name+ ".jpeg", map);}
return;
}


Err_func::Err_func(){}


float Err_func::get_error(std::string error_func){


	if(error_func == "threshold" )
		return  threshold_err(map_gt, map_pred);

	else if(error_func == "abs_rel_diff")
		return abs_rel_diff(map_gt, map_pred);

	else if(error_func == "sqr_rel_diff")
		return  sqr_rel_diff(map_gt, map_pred);

	if( error_func == "rmse_lin")
		return  rmse_lin(map_gt, map_pred);

	else if(error_func == "rmse_log")
		return  rmse_lin(map_gt, map_pred);

	else if(error_func == "rmse_log_inv")
		return  rmse_log_inv(map_gt, map_pred);

	else{
		std::cout << "ERROR -> " << error_func << " METHOD NOT VALID" << std::endl;
		return -999;
	}
}



float Err_func::threshold_err(cv::Mat map_gt, cv::Mat map_pred){

	//  Threshold error: % of y_i s.t. max(y_i/ yi* ,  y_i*/ y_i) = delta < thr

	int n =0;
	float curr_err = 0.0;

	for(int h =0; h < map_gt.rows; h++){
		for(int w =0; w < map_gt.cols; w++){

			if( (float) map_gt.at<uchar>(h,w)== (float) map_gt.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)== (float) map_pred.at<uchar>(h,w) && ((float) map_pred.at<uchar>(h,w)/ (float) map_gt.at<uchar>(h,w) < THRESHOLD_ERR) && ( (float) map_gt.at<uchar>(h,w)/ (float) map_pred.at<uchar>(h,w)< THRESHOLD_ERR) ){
				curr_err = curr_err + 1.0;
				n++;
			}
		}
	}

	return (curr_err/n);



}




float Err_func::abs_rel_diff(cv::Mat map_gt, cv::Mat map_pred){

	float curr_err = 0.0;
	float n = 0.0;

	for(int h =0; h < map_gt.rows; h++){
		for(int w =0; w < map_gt.cols; w++){

			if((float) map_gt.at<uchar>(h,w)== (float) map_gt.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)== (float) map_pred.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)){
				curr_err = curr_err + (abs( (float) map_pred.at<uchar>(h,w)*CONVERT_SCALE - (float) map_gt.at<uchar>(h,w)*CONVERT_SCALE ) / ((float) map_gt.at<uchar>(h,w)*CONVERT_SCALE));
				n++;
			}
		}
	}

	return (curr_err/n);


}


float Err_func::sqr_rel_diff(cv::Mat map_gt, cv::Mat map_pred){

	float curr_err = 0.0;
	float n = 0.0;

	for(int h =0; h < map_gt.rows; h++){
		for(int w =0; w < map_gt.cols; w++){
			if((float) map_gt.at<uchar>(h,w)== (float) map_gt.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)== (float) map_pred.at<uchar>(h,w)){
				curr_err = curr_err + ((pow((((float) map_pred.at<uchar>(h,w)*CONVERT_SCALE) - ((float) map_gt.at<uchar>(h,w)*CONVERT_SCALE) ),2))  /  ((float) map_gt.at<uchar>(h,w)*CONVERT_SCALE));
				n++;
			}
		}
	}

	return (curr_err/n);


}
float Err_func::rmse_lin(cv::Mat map_gt, cv::Mat map_pred){

	float curr_err = 0.0;
	float n = 0.0;

	for(int h =0; h < map_gt.rows; h++){
		for(int w =0; w < map_gt.cols; w++){
			if((float) map_gt.at<uchar>(h,w)== (float) map_gt.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)== (float) map_pred.at<uchar>(h,w)){
				curr_err = curr_err + (pow(   (float) map_pred.at<uchar>(h,w)*CONVERT_SCALE - (float) map_gt.at<uchar>(h,w)*CONVERT_SCALE,2));
				n++;
			}
		}
	}

	return sqrt(curr_err/n);

}



float Err_func::rmse_log(cv::Mat map_gt, cv::Mat map_pred){

	float curr_err = 0.0;
    float n = 0.0;
	for(int h =0; h < map_gt.rows; h++){
		for(int w =0; w < map_gt.cols; w++){
			if((float) map_gt.at<uchar>(h,w)== (float) map_gt.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)== (float) map_pred.at<uchar>(h,w)){
				curr_err = curr_err + (pow(   log((float) map_pred.at<uchar>(h,w)*CONVERT_SCALE) - log((float) map_gt.at<uchar>(h,w)*CONVERT_SCALE),2));
				n++;
			}

		}
	}

	return sqrt(curr_err/n);

}


float Err_func::rmse_log_inv(cv::Mat map_gt, cv::Mat map_pred){

	float partial1 = 0.0;
	float partial2 = 0.0;
	float di = 0.0;
	float n = 0.0;

	for(int h =0; h < map_gt.rows; h++){
		for(int w =0; w < map_gt.cols; w++){
			if((float) map_gt.at<uchar>(h,w)== (float) map_gt.at<uchar>(h,w) && (float) map_pred.at<uchar>(h,w)== (float) map_pred.at<uchar>(h,w)){
				di = log((float) map_pred.at<uchar>(h,w)*CONVERT_SCALE) - log((float) map_gt.at<uchar>(h,w)*CONVERT_SCALE);
				partial1 = partial1 + pow(di,2);
				partial2 = partial2 + di;
				n++;
			}
		}
	}

	return ( partial1/n - pow(partial2,2)/pow(n,2) );



}

