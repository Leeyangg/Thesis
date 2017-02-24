#include "merger.h"


cv::Point_<float> center_weight;
cv::Mat weight_mat;
std::vector<cv::Point_<int>> coordinates;	
int desire_x , desired_y;
caffe::SolverParameter solver_param;
bool on_mouse = false;

static void onMouse( int event, int x, int y, int, void* )
{	
    cv::Point_<int> curr_coor;
    curr_coor.x = x;
	 curr_coor.y = y;

	 on_mouse =true;

	 if(event =   cv::EVENT_LBUTTONDOWN ){
       coordinates.push_back(curr_coor);

	 }

    else if(event =   cv::EVENT_RBUTTONDOWN ){
	    coordinates.clear();
    }
 
}


int main(int argc, char** argv)
{
    int user_input, user_input2;
	 /*cv::namedWindow("merged",CV_WINDOW_NORMAL);
	 //cv::namedWindow("Zed Normalized",CV_WINDOW_NORMAL);
	 cv::namedWindow("Merged points",CV_WINDOW_NORMAL);
	 cv::namedWindow("Zed Stereo Depth",CV_WINDOW_NORMAL);
	 cv::namedWindow("Mono",CV_WINDOW_NORMAL);
	 cv::namedWindow("Original image",CV_WINDOW_NORMAL);
	 cv::namedWindow("Weights", CV_WINDOW_NORMAL);  
	 cv::namedWindow("Zed Stereo Depth",CV_WINDOW_NORMAL);*/
	 std::vector<cv::Mat> input_channels;
	 std::vector<cv::Mat> input_channels_lstm;
	 std::vector<cv::Mat> input_channels_eigen;
	 std::vector<cv::Mat> input_channels_fcn;
    std::string path_image;
    bool video = true;
	 int nchannels;
    cv::Mat image, image_f;
    cv::Mat depth_cnn_rescaled, depth_zed_rescaled, depth_err_rescaled;
	 float cum_err = 0.0;
	 float threshold_confidence = 0.9;
	 bool zed_input = false;
    bool vid_input = false;
    char* path_video;
    cv::VideoCapture vid;

    //Variables caffe/CNN	
    bool lstm = false;
    bool eigen = false;
    bool fcn = false;
	 bool all_cnn = false;
    Caffe::set_mode(Caffe::GPU);
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
    cv::Mat depth_cnn; 



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
	 cv::Mat depth_cnn_lstm;
	 cv::Mat depth_cnn_eigen;
	 cv::Mat depth_cnn_fcn;


    //Variables ZED	
    SENSING_MODE dm_type = RAW;
    DATA_TYPE dtype = FLOAT;
    MAT_TYPE mtype = CPU;
    sl::zed::Camera* zed;
    double min_depth_zed, max_depth_zed;
    double min_depth_err, max_depth_err;
    double min_conf, max_conf;
    float fx;
    float baseline;


    /*    
    --------------------------------------------------------------------------------------------------      
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


	 //caffe::ReadSolverParamsFromTextFileOrDie("../mix_lstm/mix_lstm_solver.prototxt", &solver_param);
   // boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

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

                if ( (file_proto = fopen("../mix_lstm/mix_lstm_deploy.prototxt", "r")) && (file_caffe_model = fopen("../mix_lstm/mix_lstm.caffemodel", "r")) && (file_train_proto = fopen("../mix_lstm/mix_lstm_train.prototxt", "r")) && (file_solver_proto = fopen("../mix_lstm/mix_lstm_solver.prototxt", "r"))) {
                    fclose(file_proto);
                    fclose(file_caffe_model);
						  fclose(file_train_proto);
						  fclose(file_solver_proto);
                    net_.reset(new Net<float>("../mix_lstm/mix_lstm_deploy.prototxt", TEST));
                    net_->CopyTrainedLayersFrom("../mix_lstm/mix_lstm.caffemodel");
                    input_layer = net_->input_blobs()[0];

						  //Setup solver
					     //dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("inputdata").get());
						  //solver->net()->ShareTrainedLayersWith(net_.get());
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


        std::cout <<  "Video -> 1 " << std::endl << "ZED -> 2" << std::endl;
        std::cin >> user_input2;

		 if( user_input2 == 1){

           std::cout <<  "Insert path to video:" << std::endl;
			  std::cin >> path_video;
			  vid_input = true;
		     vid.open(path_video);

	        if(!vid.isOpened()){
				std::cout << "FAILED OPEN VIDEO FILE'" << std::endl; 
              return -1;
           }
        }

		 else if(user_input == 2){
	     zed_input = true;
       }

		 else{
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


    /*
    --------------------------------------------------------------------------------------------------      
     Allocate memory for all the required depth maps
    --------------------------------------------------------------------------------------------------   
    */

			if(!all_cnn){
		     //Extract output
			  if(lstm || fcn)
		     	  output_norm = net_->blob_by_name("depth");   

		     else if(eigen)
		      	output_norm = net_->blob_by_name("fine_depth");  
		  
			  output_layer = net_->output_blobs()[0];
			  
			
			  output_geo = cv::Size(output_layer->width(),output_layer->height());
           depth_cnn.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1);

			}

			else{

				output_layer_lstm = net_lstm->blob_by_name("depth");
				output_layer_eigen = net_eigen->blob_by_name("fine_depth");
				output_layer_fcn = net_fcn->blob_by_name("depth");

				output_geo = cv::Size(output_layer_lstm->width(),output_layer_lstm->height());
            output_geo_eigen = cv::Size(output_layer_eigen->width(),output_layer_eigen->height());

				depth_cnn_lstm.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1);
		      depth_cnn_eigen.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1);
		      depth_cnn_fcn.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1);

			}

        //Depth Matrices
        cv::Mat un_depth(height, width, CV_32FC1);
		  cv::Mat zed_map_un_resized(output_geo.height, output_geo.width,CV_32FC1);        
        cv::Mat nn_zed_depth(height, width, CV_8UC4);
		  cv::Mat zed_map_nn_resized(output_geo.height, output_geo.width,CV_32FC1);  
		  cv::Mat depth_zed (output_geo.height, output_geo.width,CV_32FC1); 
		  cv::Mat depth_confidence(height, width, CV_32FC1);;
	     cv::Mat depth_err(output_geo.height,output_geo.width,CV_32FC1);
		  cv::Mat depth_err_m(output_geo.height,output_geo.width,CV_32FC1);
        cv::Mat depth_merged(output_geo.height,output_geo.width,CV_32FC1);
        cv::Mat merged_points(output_geo.height,output_geo.width,CV_32FC1);
	     weight_mat.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1); 





    /*
    --------------------------------------------------------------------------------------------------      
     Extract frame -> Compute depth maps -> 	Post-processing -> Plot
    --------------------------------------------------------------------------------------------------   
    */

    for(;;){
    	// cv::setMouseCallback( "Zed Stereo Depth", onMouse, 0 );  

		 if(zed_input){     
		    if (video && !zed->grab(dm_type)){

		       // Retrieve left color image
		       sl::zed::Mat left = zed->retrieveImage(sl::zed::SIDE::RIGHT);
		       imagezed = slMat2cvMat(left);
		       imagezed.copyTo(image);

		    }
       }

		if(vid_input){
			vid >> image;
		}

       if(image.cols != input_geometry_.width || image.rows != input_geometry_.height ) 
            cv::resize(image, image, input_geometry_);
      

        cv::Mat original_image(input_geometry_.height, input_geometry_.width ,CV_32FC3);         
  	     image.copyTo(original_image);
        image.convertTo(image, CV_32FC3);

        /*
        --------------------------------------------------------------------------------------------------      
         Load data
        --------------------------------------------------------------------------------------------------   
        */


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


            begin_mem_output = output_norm->cpu_data();

}
			else{
				cv::split(image, input_channels_lstm);
				cv::split(image, input_channels_eigen);
				cv::split(image, input_channels_fcn);

        CHECK(reinterpret_cast<float*>(input_channels_lstm.at(0).data)  == net_lstm->input_blobs()[0]->cpu_data()) << "LSTM Input channels are not wrapping the input layer of the network.";
        CHECK(reinterpret_cast<float*>(input_channels_eigen.at(0).data)  == net_eigen->input_blobs()[0]->cpu_data()) << "EIGEN Input channels are not wrapping the input layer of the network.";
        CHECK(reinterpret_cast<float*>(input_channels_fcn.at(0).data)  == net_fcn->input_blobs()[0]->cpu_data()) << "FCN Input channels are not wrapping the input layer of the network.";

				begin_mem_output_lstm = output_layer_lstm->cpu_data();
				begin_mem_output_eigen = output_layer_eigen->cpu_data();
				begin_mem_output_fcn = output_layer_fcn ->cpu_data();
			}




        /*
        --------------------------------------------------------------------------------------------------      
        Compute output and map to depth matrix from CNN and Zed
        --------------------------------------------------------------------------------------------------   
        */

        //Forward pass
			if(!all_cnn){
		     net_->Forward();
			}

			else{
			
				net_lstm->Forward();
				net_eigen->Forward();
				net_fcn->Forward();
			}
	
	     //Zed Unnormalized depth
        slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(un_depth);  
		  cv::resize(un_depth,zed_map_un_resized, output_geo); 

	     //Zed Normalized depth
        slMat2cvMat(zed->normalizeMeasure(sl::zed::MEASURE::DEPTH)).copyTo(nn_zed_depth);  
		  cv::resize(nn_zed_depth,zed_map_nn_resized, output_geo);

	     //Zed depth confidence map
		  slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE)).copyTo(depth_confidence); 
		  cv::resize(depth_confidence, depth_confidence, output_geo);	 		 

        for(int h=0 ; h <output_geo.height ; h++){
	
				cv::Point_<int> curr_coor;
				
            for(int w = 0 ; w < output_geo.width; w++){

					if(!all_cnn){
			      	if(eigen || fcn)
		             	depth_cnn.at<float>(h,w) = begin_mem_output[w+h*output_layer->width()]/ZED_NORMALIZATION_FACTOR;

						else
							depth_cnn.at<float>(h,w) = begin_mem_output[w+h*output_layer->width()];

						}
	
						else{

							depth_cnn_lstm.at<float>(h,w) = begin_mem_output_lstm[w+h*output_layer_lstm->width()];
							depth_cnn_fcn.at<float>(h,w) = begin_mem_output_fcn[w+h*output_layer_fcn->width()]/ZED_NORMALIZATION_FACTOR;

						  if( (h < output_geo_eigen.height) && (w < output_geo_eigen.width) )
								depth_cnn_eigen.at<float>(h,w) = begin_mem_output_eigen[w+h*output_layer_eigen->width()]/ZED_NORMALIZATION_FACTOR;
					
						}

						//if(depth_confidence.at<float>(h,w) > threshold_confidence*max_conf){
                  if(h%5 == 0 && w%5 == 0 && h > 0 && w > 0 && zed_map_un_resized.at<float>(h,w) > 0 && depth_confidence.at<float>(h,w) > threshold_confidence*max_conf){
				         curr_coor.x = w;
						   curr_coor.y = h;
						   coordinates.push_back(curr_coor);
							merged_points.at<float>(h,w) =1.0;
						}
				    
                  else{
                      merged_points.at<float>(h,w) =0.0;
						}

						//Unnormalized depth map
					   depth_zed.at<float>(h,w) = zed_map_un_resized.at<float>(h,w)*0.001;

						if(depth_zed.at<float>(h,w) > 0.0 && depth_cnn.at<float>(h,w)*ZED_NORMALIZATION_FACTOR < 6.0){
						   depth_err.at<float>(h,w) = abs( depth_zed.at<float>(h,w) - depth_cnn.at<float>(h,w)*ZED_NORMALIZATION_FACTOR) ;
						   cum_err = cum_err + depth_err.at<float>(h,w);
						}

					   //if(depth_zed.at<float>(h,w) < 0.0)
							//depth_zed.at<float>(h,w) = 1.0;

					//std::cout <<depth_cnn.at<float>(h,w) << " ";
					   }
					//std::cout << std::endl;
        } 

			if(coordinates.size() > 0){
		   	 depth_merged = merge(coordinates, depth_zed, depth_cnn, weight_mat, &center_weight);
			    circle(original_image, cv::Point_<float>(center_weight.x,  center_weight.y), 5, (255,0,0), -1, 8, 0);
			}
		
			else{
			   depth_cnn.copyTo(depth_merged);
			}

		  // if(!on_mouse)
         cum_err = 0;
         coordinates.clear();
			

/* 
		 float mean[4];

   	    mean[0] =(cv::sum(depth_zed)[0]/(output_geo.height*output_geo.width));
 mean[1] =(cv::sum(depth_cnn_lstm)[0]/(output_geo.height*output_geo.width));
 mean[2] =(cv::sum(depth_cnn_eigen)[0]/(output_geo.height*output_geo.width));
 mean[3] =(cv::sum(depth_cnn_fcn)[0]/(output_geo.height*output_geo.width));
			
			std::cout << mean[0] << " " << mean[1] << " " << mean[2] << " " << mean[3] << " " <<std::endl;*/
	
	         //dataLayer_trainnet->Reset(original_image.ptr<float>(0), mean, 1);
         //dataLayer_trainnet->Reset(im_map, mean, 1);

         //dataLayer_testnet->Reset(cop_depth.ptr<float>(0), mean, 1);
         //solver->Solve();
        //net_->ShareTrainedLayersWith(solver->net().get());

		if(!all_cnn){
			  /*
			  --------------------------------------------------------------------------------------------------      
			  Output depth map
			  --------------------------------------------------------------------------------------------------   
			  */
		     // CNN map
			  plot_maps(depth_cnn, 255*ZED_NORMALIZATION_FACTOR/NORMALIZATION_FACTOR, output_geo, cv::COLORMAP_RAINBOW, "Monocular estimation");

				//Setup ZED map
            plot_maps(depth_zed, 255/NORMALIZATION_FACTOR, output_geo, cv::COLORMAP_RAINBOW, "Stereo Estimation");

		      //Setup Merged map
            plot_maps(depth_merged, 255, output_geo, cv::COLORMAP_RAINBOW, "Merged Estimation");

				//Setup Local weight influence
            plot_maps(depth_merged, 255, output_geo, cv::COLORMAP_AUTUMN, "Local weights");
		
			    //Display other maps
				  cv::imshow("Points 4 merge", merged_points);
				 cv::imshow("Original image", original_image);
				 char key = cv::waitKey(10);
	
					switch(key){
						case 'p':
							threshold_confidence = threshold_confidence + 0.1;


						case 'm':
							threshold_confidence = threshold_confidence - 0.1;

					}

		}

		else{

			     // CNN map
				  plot_maps(depth_cnn_eigen, 255, output_geo_eigen, cv::COLORMAP_RAINBOW, "Eigen estimation");
				  plot_maps(depth_cnn_lstm,  255, output_geo,       cv::COLORMAP_RAINBOW, "LSTM estimation" );
				  plot_maps(depth_cnn_fcn,   255, output_geo,       cv::COLORMAP_RAINBOW, "FCN estimation"  );

				  //ZED map
              plot_maps(depth_zed, 255/NORMALIZATION_FACTOR, output_geo, cv::COLORMAP_RAINBOW, "Stereo Estimation");

				  //Other maps
				  cv::imshow("Original image", original_image);

				  cv::waitKey(10);

		}

	 //cv::imwrite("../images/ori.png", original_image);

        if(!video)
            break;

    }

    return 0;
}
