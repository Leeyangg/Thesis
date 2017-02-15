#include <caffe/caffe.hpp>

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


//ZED Includes
#include <zed/Camera.hpp>
#include <zed/utils/GlobalDefine.hpp>

#include "kernel.cuh"


using namespace caffe;  
using std::string;
using boost::shared_ptr;
using namespace boost::filesystem;
using namespace sl::zed;

int main(int argc, char** argv)
{
    int user_input;
    FILE *file_proto, *file_caffe_model, *file_image;
    std::string path_image;
    cv::Mat original_image;
    bool video = true;
    int nchannels;

    cv::Mat image;
    cv::Mat depth_cnn_rescaled, depth_zed_rescaled, depth_err_rescaled;

    //Variables caffe/CNN   
    bool lstm = false;
    bool eigen = false;
    bool fcn = false;
     bool all_cnn = false;
    Caffe::set_mode(Caffe::GPU);
    cv::Size input_geometry_;
    shared_ptr<Net<float> > net_;
    Blob<float>* input_layer;
    Blob<float>* output_layer;
    shared_ptr<caffe::Blob<float>> output_norm; 
    const float* begin_mem_output;
    double min_depth_cnn,  max_depth_cnn;
    float* input_data;
    cv::Size output_geo;
     cv::Size output_geo_eigen;

     //Variables all CNN at same time
    FILE *file_proto_lstm, *file_caffe_model_lstm, *file_proto_eigen, *file_caffe_model_eigen, *file_proto_fcn, *file_caffe_model_fcn ;
     shared_ptr<Net<float> > net_lstm;
    shared_ptr<Net<float> > net_eigen;
    shared_ptr<Net<float> > net_fcn;
    Blob<float>* input_layer_lstm;
     Blob<float>* input_layer_eigen;
    Blob<float>* input_layer_fcn; 
    shared_ptr<caffe::Blob<float>> output_layer_lstm;
    shared_ptr<caffe::Blob<float>> output_layer_eigen;
    shared_ptr<caffe::Blob<float>> output_layer_fcn;
    const float* begin_mem_output_lstm;
    const float* begin_mem_output_eigen;
    const float* begin_mem_output_fcn;
    float* input_data_lstm ;
     float* input_data_eigen;
     float* input_data_fcn ;
    cv::Mat depth_lstm_color, depth_eigen_color, depth_fcn_color;

    //Variables ZED 
    SENSING_MODE dm_type = FULL;
    DATA_TYPE dtype = FLOAT;
    MAT_TYPE mtype = CPU;
    sl::zed::Camera* zed;
    double min_depth_zed, max_depth_zed;
    double min_depth_err, max_depth_err;
    float fx;
    float baseline;


    /*    --------------------------------------------------------------------------------------------------      
    Setup ZED
    --------------------------------------------------------------------------------------------------   
    */  
     zed = new sl::zed::Camera(sl::zed::HD720);
    fx = zed->getParameters()->LeftCam.fx; 
    baseline = zed->getParameters()->baseline;
     zed->setDepthClampValue(20);   

    //init WITH self-calibration (- last parameter to false -)
    sl::zed::ERRCODE err = zed->init(sl::zed::MODE::PERFORMANCE, 0,true,false,false);

    std::cout <<"ErrCode : "<<sl::zed::errcode2str(err) << std::endl;

    // Quit if an error occurred
    if (err != sl::zed::SUCCESS) {
        delete zed;
        return 1;
    }

    int width = zed->getImageSize().width;
    int height = zed->getImageSize().height;
    cv::Mat imagezed(height, width, CV_32FC3,1);
    cv::Mat init_depth_zed_mat(height, width, CV_8UC4,1);

    /* 
    --------------------------------------------------------------------------------------------------      
    Setup models and image
    --------------------------------------------------------------------------------------------------   
    */  
    if (argc == 1) {
        std::cout <<  "Select setup:" << std::endl;
        std::cerr << "1: Run mix_lstm cnn " <<  std::endl
                  << "2: Run mix_eigen cnn " << std::endl
                  << "3: Run mix_fcn cnn" << std::endl
                        << "4: Run all @ same time" << std::endl;

        std::cin >> user_input;

        switch (user_input)
        {
            case 1:

                if ( (file_proto = fopen("../mix_lstm/mix_lstm_deploy.prototxt", "r")) && (file_caffe_model = fopen("../mix_lstm/mix_lstm.caffemodel", "r"))) {
                    fclose(file_proto);
                    fclose(file_caffe_model);
                    net_.reset(new Net<float>("../mix_lstm/mix_lstm_deploy.prototxt", TEST));
                    net_->CopyTrainedLayersFrom("../mix_lstm/mix_lstm.caffemodel");
                    input_layer = net_->input_blobs()[0];
                    lstm = true;
                    break;
                } 

                else {
                    std::cout << " ../mix_lstm/mix_lstm_deploy.prototxt or ../mix_lstm/mix_lstm.caffemodel does not exist" << std::endl;
                    return 0;
                }   
           
            case 2:
                if ( (file_proto = fopen("../mix_eigen/fine_net_deploy.prototxt", "r")) && (file_caffe_model = fopen("../mix_eigen/eigen_fine_mixed.caffemodel", "r"))) {
                    fclose(file_proto);
                    fclose(file_caffe_model); 
                    net_.reset(new Net<float>("../mix_eigen/fine_net_deploy.prototxt", TEST));
                    net_->CopyTrainedLayersFrom("../mix_eigen/eigen_fine_mixed.caffemodel");
                    input_layer = net_->input_blobs()[0];
                    eigen = true;
                    break;
                }

                else {
                    std::cout << " ../mix_eigen/fine_net_deploy.prototxt or ../mix_eigen/eigen_fine_mixed.caffemodel does not exist" << std::endl;
                    return 0;
                }  

            case 3:
                if ( (file_proto = fopen("../mix_fcn/mix_fcn_deploy.prototxt", "r")) && (file_caffe_model = fopen("../mix_fcn/mix_fcn.caffemodel", "r"))) {
                    fclose(file_proto);
                    fclose(file_caffe_model); 
                    net_.reset(new Net<float>("../mix_fcn/mix_fcn_deploy.prototxt", TEST));
                    net_->CopyTrainedLayersFrom("../mix_fcn/mix_fcn.caffemodel");
                    input_layer = net_->input_blobs()[0];
                    fcn = true;
                    break;
                }

                else {
                    std::cout << " ../mix_fcn/mix_fcn_deploy.prototxt or ../mix_fcn/mix_fcn.caffemodel does not exist" << std::endl;
                    return 0;
                }

                case 4: 
                    if((file_proto_lstm = fopen("../mix_lstm/mix_lstm_deploy.prototxt", "r")) && (file_caffe_model_lstm = fopen("../mix_lstm/mix_lstm.caffemodel", "r")) && (file_proto_eigen = fopen("../mix_eigen/fine_net_deploy.prototxt", "r")) && (file_caffe_model_eigen = fopen("../mix_eigen/eigen_fine_mixed.caffemodel", "r")) && (file_proto_fcn = fopen("../mix_fcn/mix_fcn_deploy.prototxt", "r")) && (file_caffe_model_fcn = fopen("../mix_fcn/mix_fcn.caffemodel", "r")) ){
                        fclose(file_proto_lstm);
                        fclose(file_caffe_model_lstm); 
                        fclose(file_proto_eigen);
                        fclose(file_caffe_model_eigen); 
                        fclose(file_proto_fcn);
                        fclose(file_caffe_model_fcn); 

                        net_lstm.reset(new Net<float>("../mix_lstm/mix_lstm_deploy.prototxt", TEST));
                        net_lstm->CopyTrainedLayersFrom("../mix_lstm/mix_lstm.caffemodel");
                        input_layer_lstm = net_lstm->input_blobs()[0];

                        net_eigen.reset(new Net<float>("../mix_eigen/fine_net_deploy.prototxt", TEST));
                        net_eigen->CopyTrainedLayersFrom("../mix_eigen/eigen_fine_mixed.caffemodel");
                        input_layer_eigen = net_eigen->input_blobs()[0];

                        net_fcn.reset(new Net<float>("../mix_fcn/mix_fcn_deploy.prototxt", TEST));
                        net_fcn->CopyTrainedLayersFrom("../mix_fcn/mix_fcn.caffemodel");
                        input_layer_fcn = net_fcn->input_blobs()[0];

                        all_cnn = true;
                        break;
                    }

                else {

                    std::cout << " Files not found" << std::endl;
                    return 0;
                }
  

            default:
                std::cerr << "  Please select a correct option " << std::endl;
                return 0;
        }

        if(!video){
                std::cout <<  "Insert path to file or -default (../images/1.png) to use default image" << std::endl;
                std::cin >> path_image;
    

            if( path_image == "-default"  ){

                 if( file_image = fopen("../images/1.png", "r") ){
                     fclose(file_image);
                     image = cv::imread("../images/1.png",CV_LOAD_IMAGE_COLOR);
                 }

                 else{
                     std::cout << "Image non existent" << std::endl;;
                     return 0;
                 }

            }
    

            else{
                const char * path2im = path_image.c_str();
                if( file_image = fopen(path2im, "r") ){
                    fclose(file_image);
                    image = cv::imread(path_image,CV_LOAD_IMAGE_COLOR);
                }
                else{
                    std::cout << "Image non existent" << std::endl;
                    return 0;
                }
            }
        }
    }

    else if (argc == 3){

        if ( (file_proto = fopen(argv[1], "r")) && (file_caffe_model = fopen(argv[2], "r"))) {

            net_.reset(new Net<float>(argv[1], TEST));
            net_->CopyTrainedLayersFrom(argv[2]);
            image = cv::imread(argv[3],CV_LOAD_IMAGE_COLOR);
            input_layer = net_->input_blobs()[0];

        }

    }

    else{

        std::cerr << "Invalid input... " << std::endl;
        std::cout << " -path_to_protofile -path_to_modelcaffe -path_to_image " <<  std::endl;
        return 0;
     
    }
    // ------------------------------------------------------------------------------------------------   
      
    /*
    --------------------------------------------------------------------------------------------------      
     Process image to be accepted
    --------------------------------------------------------------------------------------------------   
    */
    
     if(!all_cnn)
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());  

    else 
        input_geometry_ = cv::Size(input_layer_lstm->width(), input_layer_lstm->height());  


    for(;;){
           
       if (video && !zed->grab(dm_type)){

            // Retrieve right color image
            sl::zed::Mat left = zed->retrieveImage(sl::zed::SIDE::RIGHT);
            imagezed = slMat2cvMat(left);
            imagezed.copyTo(image);

       }

        if(image.cols != input_geometry_.width || image.rows != input_geometry_.height ) 
            cv::resize(image, image, input_geometry_);
               
        image.copyTo(original_image);
        image.convertTo(image, CV_32FC3);

        /*
        --------------------------------------------------------------------------------------------------      
         Load data
        --------------------------------------------------------------------------------------------------   
        */
        std::vector<cv::Mat> input_channels;
        std::vector<cv::Mat> input_channels_lstm;
        std::vector<cv::Mat> input_channels_eigen;
        std::vector<cv::Mat> input_channels_fcn;

        if(!all_cnn){
            input_data = input_layer->mutable_cpu_data();
            nchannels = input_layer->channels();
        }

        else{

            input_data_lstm = input_layer_lstm->mutable_cpu_data();
            input_data_eigen = input_layer_eigen->mutable_cpu_data();
            input_data_fcn = input_layer_fcn->mutable_cpu_data();

            nchannels = input_layer_lstm->channels();
        }

        for (int i = 0; i < nchannels; ++i) {

            if(!all_cnn){
                cv::Mat channel(input_geometry_.height,input_geometry_.width, CV_32FC1, input_data);
                input_channels.push_back(channel);
                input_data += input_geometry_.width * input_geometry_.height;
            }

            else{

                cv::Mat channel_lstm(input_geometry_.height,input_geometry_.width, CV_32FC1, input_data_lstm);
                input_channels_lstm.push_back(channel_lstm);
                input_data_lstm += input_geometry_.width * input_geometry_.height;

                cv::Mat channel_eigen(input_geometry_.height,input_geometry_.width, CV_32FC1, input_data_eigen);
                input_channels_eigen.push_back(channel_eigen);
                input_data_eigen += input_geometry_.width * input_geometry_.height;

                cv::Mat channel_fcn(input_geometry_.height,input_geometry_.width, CV_32FC1, input_data_fcn);
                input_channels_fcn.push_back(channel_fcn);
                input_data_fcn += input_geometry_.width * input_geometry_.height;

            }
        }

        if(!all_cnn){
        cv::split(image, input_channels);
        CHECK(reinterpret_cast<float*>(input_channels.at(0).data)  == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";

        }

        else{

           cv::split(image, input_channels_lstm);
           cv::split(image, input_channels_eigen);
           cv::split(image, input_channels_fcn);

           CHECK(reinterpret_cast<float*>(input_channels_lstm.at(0).data)  == net_lstm->input_blobs()[0]->cpu_data()) << "LSTM Input channels are not wrapping the input layer of the network.";
           CHECK(reinterpret_cast<float*>(input_channels_eigen.at(0).data)  == net_eigen->input_blobs()[0]->cpu_data()) << "EIGEN Input channels are not wrapping the input layer of the network.";
           CHECK(reinterpret_cast<float*>(input_channels_fcn.at(0).data)  == net_fcn->input_blobs()[0]->cpu_data()) << "FCN Input channels are not wrapping the input layer of the network."

        }



        /*
        --------------------------------------------------------------------------------------------------      
        Compute output and map to depth matrix from CNN and Zed
        --------------------------------------------------------------------------------------------------   
        */

        //Forward pass
        if(!all_cnn){
           net_->Forward();

           //Extract output
           if(lstm || fcn)
               output_norm = net_->blob_by_name("depth");   

           else if(eigen)
               output_norm = net_->blob_by_name("fine_depth");  
      
            output_layer = net_->output_blobs()[0];
            begin_mem_output = output_norm->cpu_data();
        
            output_geo = cv::Size(output_layer->width(),output_layer->height());

        }


        else{
        
            net_lstm->Forward();
            net_eigen->Forward();
            net_fcn->Forward();

            output_layer_lstm = net_lstm->blob_by_name("depth");
            output_layer_eigen = net_eigen->blob_by_name("fine_depth");
            output_layer_fcn = net_fcn->blob_by_name("depth");

            begin_mem_output_lstm = output_layer_lstm->cpu_data();
            begin_mem_output_eigen = output_layer_eigen->cpu_data();
            begin_mem_output_fcn = output_layer_fcn ->cpu_data();

            output_geo = cv::Size(output_layer_lstm->width(),output_layer_lstm->height());
            output_geo_eigen = cv::Size(output_layer_eigen->width(),output_layer_eigen->height());

        }
    
        //Zed Dense Depth map
        cv::Mat un_depth(height, width, CV_32FC1);
        slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(un_depth);  

        cv::Mat nn_zed_depth(height, width, CV_8UC4);
        slMat2cvMat(zed->normalizeMeasure(sl::zed::MEASURE::DEPTH)).copyTo(nn_zed_depth);  

        cv::Mat zed_map_resized;        
        cv::resize(un_depth,zed_map_resized, output_geo);

        //Depth Matrices
        cv::Mat depth_cnn(output_geo.height, output_geo.width,CV_32FC1); 
        cv::Mat depth_cnn_lstm(output_geo.height, output_geo.width,CV_32FC1);
        cv::Mat depth_cnn_eigen(output_geo_eigen.height, output_geo_eigen.width,CV_32FC1);
        cv::Mat depth_cnn_fcn(output_geo.height, output_geo.width,CV_32FC1);
        cv::Mat depth_zed(output_geo.height, output_geo.width,CV_32FC1);
        cv::Mat depth_zed_un(output_geo.height, output_geo.width,CV_32FC1);
        cv::Mat depth_err(output_geo.height,output_geo.width,CV_32FC1);

        for(int h=0 ; h <output_geo.height ; h++){
            for(int w = 0 ; w < output_geo.width; w++){
                    if(!all_cnn){

                        if(eigen || fcn)
                            depth_cnn.at<float>(h,w) = begin_mem_output[w+h*output_layer->width()]/39.75;

                        else
                            depth_cnn.at<float>(h,w) = begin_mem_output[w+h*output_layer->width()];

                        depth_err.at<float>(h,w) = abs( depth_zed_un.at<float>(h,w) - depth_cnn.at<float>(h,w)*39.75) ;

                    }
    
                    else{

                        depth_cnn_lstm.at<float>(h,w) = begin_mem_output_lstm[w+h*output_layer_lstm->width()];
                        depth_cnn_fcn.at<float>(h,w) = begin_mem_output_fcn[w+h*output_layer_fcn->width()]/39.75;

                        if( (h < output_geo_eigen.height) && (w < output_geo_eigen.width) )
                           depth_cnn_eigen.at<float>(h,w) = begin_mem_output_eigen[w+h*output_layer_eigen->width()]/39.75;
                
                    }

                    //Unnormalized depth map
                    depth_zed.at<float>(h,w) = zed_map_resized.at<float>(h,w)*0.001;

                    if(depth_zed.at<float>(h,w) < 0)
                        depth_zed.at<float>(h,w) = 255;

                    //std::cout <<depth_cnn_lstm.at<float>(h,w) << " ";
            }
                    //std::cout << std::endl;
        }   

        if(!all_cnn){
            /*
            --------------------------------------------------------------------------------------------------      
            Output depth map
            --------------------------------------------------------------------------------------------------   
            */

            //Setup CNN map
            cv::minMaxIdx(depth_cnn, &min_depth_cnn, &max_depth_cnn);
            cv::convertScaleAbs(depth_cnn, depth_cnn_rescaled, 255);              
            cv::Mat cnn_colour_depth(output_layer->height(), output_layer->width(), CV_32FC3);
            applyColorMap(depth_cnn_rescaled, cnn_colour_depth, cv::COLORMAP_RAINBOW);

            //Setup ZED map
            cv::minMaxIdx(depth_zed, &min_depth_zed, &max_depth_zed);
            cv::convertScaleAbs(depth_zed,depth_zed_rescaled, 255/39.75);
            cv::Mat zed_colour_depth(output_layer->height(), output_layer->width(), CV_32FC3);
            applyColorMap(depth_zed_rescaled,zed_colour_depth,cv::COLORMAP_RAINBOW);

            //Setup error map
            cv::minMaxIdx(depth_err, &min_depth_err, &max_depth_err);
            cv::convertScaleAbs(depth_err,depth_err_rescaled, 255 /max_depth_err);
            cv::Mat depth_err_color(output_layer->height(), output_layer->width(), CV_32FC3);
            applyColorMap(depth_err_rescaled, depth_err_color, cv::COLORMAP_RAINBOW);

            //Display
            cv::imshow("Source mono", depth_cnn);
            cv::imshow("Error", depth_err_color);
            cv::imshow("Zed Stereo Depth",zed_colour_depth);
            cv::imshow("Mono", cnn_colour_depth);
            cv::imshow("Original image", original_image);
            cv::waitKey(10);

        }

        else{

    
            cv::resize(nn_zed_depth, depth_zed_rescaled, output_geo);
            // applyColorMap( depth_cnn_lstm,depth_lstm_color, cv::COLORMAP_RAINBOW);
            cv::imshow("Zed",  depth_zed_rescaled);

            cv::convertScaleAbs( depth_cnn_eigen,  depth_cnn_eigen, 255);
            applyColorMap( depth_cnn_eigen, depth_eigen_color, cv::COLORMAP_RAINBOW);
            cv::imshow("Eigen", depth_eigen_color);

            cv::convertScaleAbs(depth_cnn_lstm, depth_cnn_lstm, 255);
            applyColorMap( depth_cnn_lstm, depth_lstm_color, cv::COLORMAP_RAINBOW);
            cv::imshow("LSTM",depth_lstm_color);

            cv::convertScaleAbs(depth_cnn_fcn, depth_cnn_fcn, 255);
            applyColorMap( depth_cnn_fcn, depth_fcn_color, cv::COLORMAP_RAINBOW);
            cv::imshow("FCN", depth_fcn_color);

            cv::imshow("Original image", original_image);

            cv::waitKey(10);

        }

    // cv::imwrite("../images/final_depth.png", final_depth);

        if(!video)
            break;

    }

    return 0;
}
