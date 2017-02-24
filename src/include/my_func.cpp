#include "my_func.h"

cv::Mat merge(std::vector<cv::Point_<int>> coord, cv::Mat stereo, cv::Mat mono,cv::Mat weight_mat , cv::Point_<float>* center_weight ){

	cv::Mat merged(stereo.rows, stereo.cols,CV_32FC1);
	cv::Mat dx;
   cv::Mat dy;
	std::vector<float>  wnn;
   float w1, w2, w3, w4, min_wnn =0.0, sum_w=0.0, inter_depth, w_norm;
	float sig1 = 15.0;
	float sig2 = 0.1;
	float sig3 = 1*exp(-3);
   bool first = true;
  
   cv::convertScaleAbs( mono,  mono, ZED_NORMALIZATION_FACTOR);

   cv::Sobel(mono,dx, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
   cv::Sobel(mono,dy, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT); 

   center_weight->x = coord[0].x ;
	center_weight->y = coord[0].y ;

	for(int h=0 ; h <mono.rows ; h++){

	  for(int w = 0 ; w < mono.cols; w++){

		  if(stereo.at<float>(h,w) > 0.0){

	     for(int i = 0; i < coord.size(); i++){

		     w1 = exp( (-1*sqrt(  pow((h - coord[i].y),2) +  pow((w - coord[i].x),2) ))/sig1 );
			  w2 = (1/(abs(dx.at<float>(coord[i].y, coord[i].x)- dx.at<float>(h,w)) + sig2)) * (1/(abs(   dy.at<float>(coord[i].y, coord[i].x)- dy.at<float>(h,w)  ) + sig2)) ;
           w3 = exp( -abs( mono.at<float>(h,w)  + dx.at<float>(h,w)*(h - coord[i].y) - mono.at<float>(coord[i].y, coord[i].x)  ) ) + sig3;
           w4 = exp( -abs( mono.at<float>(h,w)  + dy.at<float>(h,w)*(h - coord[i].y) - mono.at<float>(coord[i].y, coord[i].x)  ) ) + sig3;
           
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

	     for(int ii = 0; ii < coord.size(); ii++){
   
         w_norm = (wnn[ii] - min_wnn)/(sum_w - min_wnn);
		   inter_depth = inter_depth +  w_norm*( stereo.at<float>(coord[ii].y,coord[ii].x) + mono.at<float>(h,w) - mono.at<float>(coord[ii].y, coord[ii].x) );

             if(ii == 0 )
				 	  weight_mat.at<float>(h,w) = w_norm;		

		  }

	     merged.at<float>(h,w) =  inter_depth/NORMALIZATION_FACTOR;
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
			else
			merged.at<float>(h,w) =  mono.at<float>(h,w);
	  }

	}

      return merged;
}


void plot_maps(cv::Mat map, float scale_factor, cv::Size geometry, int color_map, const char* name_window){

	cv::Mat map_color(geometry.height, geometry.width, CV_32FC3);
	
   cv::convertScaleAbs(map, map, scale_factor);
	
	if(color_map != -1){

		applyColorMap(map, map_color, color_map);

		cv::imshow(name_window, map_color);
   }

   else
		cv::imshow(name_window, map);

}


