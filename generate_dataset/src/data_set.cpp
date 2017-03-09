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
#include <iostream>
#include <fstream>



int main(int argc, char** argv)
{

    cv::Mat image;
    cv::Mat image_resize;
    int rows = 40, cols = 64;
    cv::Mat im_float;
    int frame = 1;
    std::string path_frame, labels_file;
    cv::Size input_geometry_(cols, rows);
    float av_depth;
    std::ofstream labels;
    bool coloured;
    float depth;


    if(argc == 2 && (atoi(argv[1]) == 1 || atoi(argv[1]) == 3)){ 
         if(atoi(argv[1]) == 1 )
            coloured = false;

        else
            coloured = true;
    }

    else{
        std::cout << "Incorrect input";
        return 0;
    }

    std::cout <<  "Insert path to image and depths directory:" << std::endl;
    std::cin >> path_frame;

    labels_file = path_frame + "labels.txt";
    labels.open(labels_file);

    for(;;){

        av_depth = 0.0;
        image = cv::imread(path_frame + std::to_string(frame)+ ".png", coloured);


        if(! image.data )                             
        {
            std::cout <<  "End of files" << std::endl ;
            return -1;
        }

        if(image.cols != cols || image.rows != rows ){
            cv::resize(image, image_resize, input_geometry_);
            cv::imwrite(path_frame + std::to_string(frame)+ ".png", image_resize);
        }

        else
            image.copyTo(image_resize);

        // Compute depth average
        if(!coloured){
            image_resize.convertTo(im_float, CV_32FC1);
            int aux = 0;
            for(int h =0; h < rows; h++){
                for(int w =0; w < cols; w++){
                    if( im_float.at<float>(h,w) > 0.0 )
                        av_depth = av_depth + im_float.at<float>(h,w)*10000.0/(255*1000.0);
                        aux++;
                }
            }

            av_depth = av_depth /(aux);

           labels << "labels/" + std::to_string(frame)+ ".png " << av_depth << '\n'; 
        }
        std::cout << frame << " ";
        frame++;
    }

    labels.close();

    return 0;

}
// /home/diogo/Desktop/datasets/copy/test_gt/
// /home/diogo/Desktop/datasets/copy/train_gt/
