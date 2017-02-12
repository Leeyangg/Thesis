#include <caffe/caffe.hpp>
#include <memory>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using boost::shared_ptr;
using namespace boost::filesystem;

int main(int argc, char** argv)
{
    int user_input;
    FILE *file_proto, *file_caffe_model, *file_image;
    std::string path_image;
    cv::Mat original_image, image;
    bool lstm = false;
    bool eigen = false;
    bool fcn = false;
    Caffe::set_mode(Caffe::CPU);
    cv::Size input_geometry_;
    shared_ptr<Net<float> > net_;
    Blob<float>* input_layer;
    Blob<float>* output_layer; 
    const float* begin_mem_output;

    /*
    --------------------------------------------------------------------------------------------------      
    Setup models and image
    --------------------------------------------------------------------------------------------------   
    */  
    if (argc == 1) {
        std::cout <<  "Select setup:" << std::endl;
        std::cerr << "1: Run mix_lstm cnn " <<  std::endl
                  << "2: Run mix_eigen cnn " << std::endl
                  << "3: Run mix_fcn cnn" << std::endl;

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

            default:
                std::cerr << "  Please select a correct option " << std::endl;
                return 0;
        }

        std::cout <<  "Insert path to file or -default (../images/1.png) to use default image" << std::endl;
        std::cin >> path_image;

        if( path_image == "-default"  ){
            if( file_image = fopen("../images/zur.png", "r") ){
                fclose(file_image);
                image = cv::imread("../images/zur.png",CV_LOAD_IMAGE_COLOR);
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



    
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    bool video = false;
    cv::VideoCapture cap; 

    if(video){
        cap.open(0); 
        if(!cap.isOpened()) 
            return -1;
    }

    for(;;)
    {
     
        if(video)    
            cap >> image; 

        if(lstm == 1 || eigen == 1){

            if(image.cols != input_geometry_.width || image.rows != input_geometry_.height ) 
               cv::resize(image, image, input_geometry_);
                image.copyTo(original_image);

        }

        else{

            cv::resize(image, image, input_geometry_);
            image.copyTo(original_image);
            net_->Reshape();

         }

        image.convertTo(image, CV_32FC3);

        /*
        --------------------------------------------------------------------------------------------------      
         Load data
        --------------------------------------------------------------------------------------------------   
        */
        std::vector<cv::Mat> input_channels;
        float* input_data = input_layer->mutable_cpu_data();

        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(input_layer->height(),input_layer->width(), CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += input_layer->width() * input_layer->height();
        }

        cv::split(image, input_channels);
        CHECK(reinterpret_cast<float*>(input_channels.at(0).data)  == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";


        /*
        --------------------------------------------------------------------------------------------------      
        Compute output and map to depth matrix
        --------------------------------------------------------------------------------------------------   
        */
        //Forward pass
        net_->Forward();

        //Extract output
        /* Copy the output layer to a std::vector */
        output_layer = net_->output_blobs()[0];
        begin_mem_output = output_layer->cpu_data();

        //Show output
        //std::cout << output_layer->num() << " " <<  output_layer->channels() << " " << output_layer->height() << " " << output_layer->width() << " "<< std::endl;
        //std::cout << output.size() << " "<< std::endl;
        cv::Mat depth_map(output_layer->height(), output_layer->width(),CV_32FC1); 

        for(int h=0 ; h < output_layer->height() ; h++){
            for(int w = 0 ; w < output_layer->width(); w++){
                depth_map.at<float>(h,w) =  begin_mem_output[w+h*output_layer->width()];
                //std::cout << depth_map.at<float>(h,w)  << std::endl;
            }
        }   


        /*
        --------------------------------------------------------------------------------------------------      
        Output depth map
        --------------------------------------------------------------------------------------------------   
        */

        double min;
        double max;
        cv::minMaxIdx(depth_map, &min, &max);
        //cv::Mat adjMap;
        //cv::convertScaleAbs(depth_map, adjMap, 255 /max);
        
        std::cout << max;

        float tempDepth;                  // Temporary storage of depth to convert to color
        int depthRed;                   // Red value for a given depth
        int depthGreen;                 // Green value for a given depth
        int depthBlue;                  // Blue value for a given depth
       
        cv::Mat final_depth(output_layer->height(), output_layer->width(), CV_32FC3);
        for (int row = 0; row < output_layer->height(); row++) {
            for (int col = 0; col < output_layer->width(); col++) {
                tempDepth = depth_map.at<float>(row,col)*255/40;
                //std::cout << tempDepth << " ";
                //std::cout << depth_map.at<float>(row,col) << " " <<  adjMap.at<float>(row,col) << " // ";
                if(tempDepth < 43){
                    depthRed = tempDepth * 6;
                    depthGreen = 0;
                    depthBlue = tempDepth * 6;
                }
                if(tempDepth > 42 && tempDepth < 85){
                    depthRed = 255 - (tempDepth - 43) * 6;
                    depthGreen = 0;
                    depthBlue = 255;
                }
                if(tempDepth > 84 && tempDepth < 128){
                    depthRed = 0;
                    depthGreen = (tempDepth - 85) * 6;
                    depthBlue = 255;
                }
                if(tempDepth > 127 && tempDepth < 169){
                    depthRed = 0;
                    depthGreen = 255;
                    depthBlue = 255 - (tempDepth - 128) * 6;
                }
                if(tempDepth > 168 && tempDepth < 212){
                    depthRed = (tempDepth - 169) * 6;
                    depthGreen = 255;
                    depthBlue = 0;
                }
                if(tempDepth > 211 && tempDepth < 254){
                    depthRed = 255;
                    depthGreen = 255 - (tempDepth - 212) * 6;
                    depthBlue = 0;
                }
                if(tempDepth > 253){
                    depthRed = 255;
                    depthGreen = 0;
                    depthBlue = 0;
                }

                cv::Vec3f intensity(depthBlue, depthGreen, depthRed); 
                final_depth.at<cv::Vec3f>(row, col)= intensity;

            }
        }

        cv::imshow("Depth Map", final_depth);
        cv::imwrite("../images/final_depth.png", final_depth);
        cv::imshow("Original image", original_image);
        cv::waitKey(10);

        if(!video)
            break;

    }

    return 0;
}
