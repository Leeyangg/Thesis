#include "merger.h"


int main(int argc, char** argv)
{
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
            	 path_to_protofile  = "../mix_lstm/mix_lstm_deploy.prototxt";
            	 path_to_modelcaffe = "../mix_lstm/mix_lstm.caffemodel";
                 lstm = true;  
                break;
           
            case 2:
                path_to_protofile  = "../mix_eigen/fine_net_deploy.prototxt";
            	path_to_modelcaffe = "../mix_eigen/eigen_fine_mixed.caffemodel";
                eigen = true;
                break;
          
            case 3:
                path_to_protofile  = "../mix_fcn/mix_fcn_deploy.prototxt";
            	path_to_modelcaffe = "../mix_fcn/mix_fcn.caffemodel";
                fcn = true;

            default:
                std::cerr << "  Please select a correct option " << std::endl;
                return 0;
        }


        std::cout <<  "Video -> 1 " << std::endl << "ZED -> 2" << std::endl << "Image Sequence -> 3"  << std::endl;
        std::cin.sync();
        std::cin >> user_input;

		 if( user_input == 1){

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

		    #ifdef COMPILE_ZED
		    	zed_input = true;

		    #else
		    	std::cout << "ERROR: ZED NOT FOUND" << std::endl;
		    	return 0;

		     #endif
       }

		 else if(user_input == 3){
            std::cout <<  "Insert path to PNG image directory:" << std::endl;
			std::cin >> path_frame;
 			std::cin.sync();

            std::cout <<  "Insert path to PNG depth directory:" << std::endl;
			std::cin >> path_depth;
 			std::cin.sync();

            std::cout <<  "Insert sequence log title:" << std::endl;
			std::cin >> images_path_format;
			im_seq = true;


       }

		 else{
                std::cerr << "  Please select a correct option " << std::endl;
                return 0;
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

    /*
    --------------------------------------------------------------------------------------------------      
    Setup ZED
    --------------------------------------------------------------------------------------------------   
    */  
	#ifdef COMPILE_ZED

		zed = new sl::zed::Camera(sl::zed::HD720);
	    fx = zed->getParameters()->LeftCam.fx; 
	    baseline = zed->getParameters()->baseline;
		zed->setDepthClampValue(MAX_DEPTH_ZED);	

	    //init WITH self-calibration (- last parameter to false -)
	    sl::zed::ERRCODE err = zed->init(sl::zed::MODE::PERFORMANCE, 0,true,false,false);

	    std::cout <<"ErrCode : "<<sl::zed::errcode2str(err) << std::endl;

	    // Quit if an error occurred
		if (err != sl::zed::SUCCESS) {
	        delete zed;
	        return 1;
	    }

	    width = zed->getImageSize().width;
	    height = zed->getImageSize().height;
	    cv::Mat imagezed(height, width, CV_32FC3,1);
	    cv::Mat init_depth_zed_mat(height, width, CV_8UC4,1);

	#else

	    image = cv::imread(path_frame + images_path_format + "1.png", 1);
	    if(! image.data ){
			std::cout <<  " Could not open or find image" << std::endl ;
			return -1;
		}

		height = image.rows;
		width = image.cols;

	#endif

      /*
    --------------------------------------------------------------------------------------------------      
     Setup CNN
    --------------------------------------------------------------------------------------------------   
    */  
  
    if ( (file_proto = fopen(path_to_protofile.c_str(), "r")) && (file_caffe_model = fopen(path_to_modelcaffe.c_str(), "r"))  ){
                    fclose(file_proto);
                    fclose(file_caffe_model);
                    net_.reset(new Net<float>(path_to_protofile.c_str(), TEST));
                    net_->CopyTrainedLayersFrom(path_to_modelcaffe.c_str());
                    input_layer = net_->input_blobs()[0];              
   					input_geometry_ = cv::Size(input_layer->width(), input_layer->height()); 
    } 

    else {
	    std::cout << path_to_protofile << "or" << path_to_modelcaffe <<  "does not exist" << std::endl;
	    return 0;
	}  


    /*
    --------------------------------------------------------------------------------------------------      
     Allocate memory for all the required depth maps
    --------------------------------------------------------------------------------------------------   
    */

	if(lstm || fcn)
      output_norm = net_->blob_by_name("depth");   

    else if(eigen)
    	output_norm = net_->blob_by_name("fine_depth");  
  
  	output_layer = net_->output_blobs()[0];
  	
	output_geo = cv::Size(output_layer->width(),output_layer->height());
	depth_cnn.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1);
  	depth_cnn_un.create(cv::Size(output_geo.width,output_geo.height),CV_32FC1);

    //Depth Matrices
	cv::Mat un_depth(height, width, CV_32FC1);
    cv::Mat nn_zed_depth(height, width, CV_8UC4);
	cv::Mat zed_map_un_resized(output_geo.height, output_geo.width,CV_32FC1);        
	cv::Mat n_zed_depth_im_seq(height, width, CV_8UC1);
    cv::Mat n_zed_depth_im_seq_resized(output_geo.height, output_geo.width, CV_8UC1);
	cv::Mat zed_map_nn_resized(output_geo.height, output_geo.width,CV_32FC1);  
	cv::Mat depth_zed(output_geo.height, output_geo.width,CV_32FC1); 
	cv::Mat depth_confidence(height, width, CV_32FC1);
    depth_confidence.create(output_geo.height, output_geo.width, CV_32FC1);
	cv::Mat depth_conf_im_seq(height, width, CV_8UC1);
    cv::Mat depth_conf_im_seq_resized(output_geo.height, output_geo.width, CV_8UC1);
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
   
    	start = std::clock();
	    frame_counter++;
        //std::cout << frame_counter << std::endl;
		#ifdef COMPILE_ZED
			if(zed_input){     
			    if (zed->grab(dm_type)){
			       // Retrieve left color image
			       sl::zed::Mat left = zed->retrieveImage(sl::zed::SIDE::RIGHT);
			       imagezed = slMat2cvMat(left);
			       imagezed.copyTo(image);
			    }
	        }
		#endif

		if(vid_input){
			vid >> image;
		}

 		if(im_seq){

			//image = cv::imread("/home/diogo/Desktop/datasets/copy/train/labels/" + std::to_string(frame_counter)+ ".png", 1);
	    image = cv::imread(path_frame + images_path_format + std::to_string(frame_counter)+ ".png", 1);

			if(! image.data ){
					  std::cout <<  " Could not open or find image" << std::endl ;
					  return -1;
				 }
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

		input_data = input_layer->mutable_cpu_data();
		nchannels = input_layer->channels();
		
        for (int i = 0; i < nchannels; ++i) {
		         cv::Mat channel(input_geometry_.height,input_geometry_.width, CV_32FC1, input_data);
		         input_channels.push_back(channel);
		         input_data += input_geometry_.width * input_geometry_.height;
        }

     	cv::split(image, input_channels);
        CHECK(reinterpret_cast<float*>(input_channels.at(0).data)  == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
        begin_mem_output = output_norm->cpu_data();

        /*
        --------------------------------------------------------------------------------------------------      
        Compute output and map to depth matrix from CNN and Zed
        --------------------------------------------------------------------------------------------------   
        */

        //Forward pass
		 net_->Forward();

		if(!im_seq){

			#ifdef COMPILE_ZED
				//Zed Unnormalized depth
			    slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(un_depth);  
				cv::resize(un_depth,zed_map_un_resized, output_geo); 

				//Zed Normalized depth
			    slMat2cvMat(zed->normalizeMeasure(sl::zed::MEASURE::DEPTH)).copyTo(nn_zed_depth);  
				cv::resize(nn_zed_depth,zed_map_nn_resized, output_geo);

				//Zed depth confidence map
				slMat2cvMat(zed->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE)).copyTo(depth_confidence); 
				cv::resize(depth_confidence, depth_confidence, output_geo);	 

	            cv::resize(depth_confidence, depth_confidence, output_geo);	
	            cv::minMaxLoc(depth_confidence, &min_conf, &max_conf);

			#else

	            int aaaa = 0;

			#endif		 
        }

		else{
			
			//n_zed_depth_im_seq =  cv::imread("/home/diogo/Desktop/datasets/copy/train_gt/labels/" + std::to_string(frame_counter)+ ".png", 1);
			n_zed_depth_im_seq =  cv::imread(path_depth + images_path_format + std::to_string(frame_counter)+ ".png", 1);

		    if(!n_zed_depth_im_seq.data ){
				std::cout <<  "Could not open or find the image" << std::endl ;
				return -1;
			}

			cv::resize(n_zed_depth_im_seq,n_zed_depth_im_seq_resized, output_geo);

			if(IMPORT_CONFIDENCE){
		    	depth_conf_im_seq =  cv::imread("/home/diogo/Desktop/datasets/mine/depth_maps/1/depths/confidence/20000_f" + std::to_string(frame_counter)+ ".png", 1);

		    	if(!depth_conf_im_seq.data ){
					std::cout <<  "Could not open or find confidence map" << std::endl ;
					return -1;
				}

				cv::resize(depth_conf_im_seq,depth_conf_im_seq_resized, output_geo);
			}
		 }

		float mean_cnn=0.0, mean_zed =0.0;	
        int aux = 0;	

        for(int h=0 ; h <output_geo.height ; h++){
	
			cv::Point_<int> curr_coor;
				
            for(int w = 0 ; w < output_geo.width; w++){

			    if(eigen || fcn)
		            depth_cnn.at<float>(h,w) = begin_mem_output[w+h*output_layer->width()]/CNN_NORMALIZATION_FACTOR;

				else
					depth_cnn.at<float>(h,w) = begin_mem_output[w+h*output_layer->width()];

					depth_cnn_un.at<float>(h,w) = depth_cnn.at<float>(h,w)*CNN_NORMALIZATION_FACTOR;
				
						//if(depth_confidence.at<float>(h,w) > threshold_confidence*max_conf){
              /*    if(h%10 == 0 && w%10 == 0 && h > 0 && w > 0  && zed_map_un_resized.at<float>(h,w) > 0.0 && depth_confidence.at<float>(h,w) > THRESHOLD_CONFIDENCE*max_conf){
				         curr_coor.x = w;
						   curr_coor.y = h;
						   coordinates.push_back(curr_coor);
							merged_points.at<float>(h,w) = 1.0;
						}
				    
                  else{
                      merged_points.at<float>(h,w) = 0.0;
						}
*/


				//Unnormalized depth map
				if(zed_input)
					depth_zed.at<float>(h,w) = zed_map_un_resized.at<float>(h,w)*0.001;

				else{
					 depth_zed.at<float>(h,w) = (float)  n_zed_depth_im_seq_resized.data[h*n_zed_depth_im_seq_resized.step + w*n_zed_depth_im_seq_resized.elemSize()]*10000.0/(255*1000.0); 

					if(IMPORT_CONFIDENCE)
			        	depth_confidence.at<float>(h,w)  = (float) depth_conf_im_seq_resized.data[h*depth_conf_im_seq_resized.step + w*n_zed_depth_im_seq_resized.elemSize()]; 
				}

				//std::cout << depth_cnn.at<float>(h,w) << " ";

				if(depth_confidence.at<float>(h,w) < 2.1 && zed_input)
					depth_zed.at<float>(h,w) = 200;

				if(!zed_input && IMPORT_CONFIDENCE){
					if(depth_confidence.at<float>(h,w) < 2.1)
						depth_zed.at<float>(h,w) = 200;
				}

						/*if(depth_zed.at<float>(h,w) > 0.0 && depth_cnn.at<float>(h,w)*CNN_NORMALIZATION_FACTOR < 6.0){
						   depth_err.at<float>(h,w) = abs( depth_zed.at<float>(h,w) - depth_cnn.at<float>(h,w)*CNN_NORMALIZATION_FACTOR) ;
						   cum_err = cum_err + depth_err.at<float>(h,w);
						}*/

				if(depth_zed.at<float>(h,w) > 0.0 && depth_cnn_un.at<float>(h,w) > 0.0){
					val_points++; 
				}		

				mean_cnn = mean_cnn + depth_cnn.at<float>(h,w);					

					   //if(depth_zed.at<float>(h,w) < 0.0)
							//depth_zed.at<float>(h,w) = 1.0;

			}
					//std::cout << val_points << std::endl;
        } 

		if(coordinates.size() > 0){
	   	// depth_merged = merge(coordinates, depth_zed, depth_cnn_un, weight_mat, &center_weight);
		    circle( merged_points, cv::Point_<float>(center_weight.x,  center_weight.y), 5, (255,255,255), -1, 8, 0);
		}
		
		else{
		   depth_cnn.copyTo(depth_merged);
		}

		errors.push_mat_gt( depth_zed);
		errors.push_mat_pred( depth_cnn_un);
        std::cout << "threshold_err = " << errors.get_error("threshold") << std::endl;
	    std::cout << "abs_rel_diff error = " << errors.get_error("abs_rel_diff") << std::endl;
        std::cout << "sqr_rel_diff = " << errors.get_error("sqr_rel_diff") << std::endl;
	    std::cout << "rmse_lin = " << errors.get_error("rmse_lin") << std::endl;
	    std::cout << "rmse_log = " << errors.get_error("rmse_log") << std::endl;
        std::cout << "inv scale error = " << errors.get_error("rmse_log_inv") << std::endl << std::endl;
		std::cout << val_points << "/" << merged_points.rows*merged_points.cols << " valid points" << std::endl;


		 // if(!on_mouse)
        cum_err = 0;
	    val_points = 0;
        coordinates.clear();
		mean_cnn = CNN_NORMALIZATION_FACTOR*mean_cnn/(output_geo.height*output_geo.width);
        mean_zed = mean_zed/(aux);

			//std::cout << "Mean ZED = " << mean_zed << "     Mean CNN = " << mean_cnn << std::endl;


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

			  /*
			  --------------------------------------------------------------------------------------------------      
			  Output depth map
			  --------------------------------------------------------------------------------------------------   
			  */

		     // CNN map
	    if(display == 1)
	  		plot_maps(depth_cnn, 255*CNN_NORMALIZATION_FACTOR/NORMALIZATION_FACTOR, output_geo, cv::COLORMAP_RAINBOW, act_window, save_image);

		//Setup ZED map
	 	if(display == 2 && !im_seq)
   			plot_maps(depth_zed, 255/NORMALIZATION_FACTOR, output_geo, cv::COLORMAP_RAINBOW, act_window, save_image);

		if(display == 3)
    		plot_maps(depth_zed, 255.0/10.0, output_geo, cv::COLORMAP_RAINBOW, act_window, save_image);

	 	if(display == 2 && im_seq)
   			plot_maps(n_zed_depth_im_seq_resized, 1, output_geo, cv::COLORMAP_RAINBOW, act_window, save_image);

      //Setup Merged map
		if(display == 4)
    		plot_maps(depth_merged, 255, output_geo, cv::COLORMAP_RAINBOW, act_window, save_image);

		if(display == 5)
    		plot_maps(depth_merged, 255*NORMALIZATION_FACTOR/10, output_geo, cv::COLORMAP_RAINBOW, act_window, save_image);

		//Setup Local weight influence

		if(display == 6)
    		plot_maps(weight_mat, 255, output_geo, cv::COLORMAP_BONE, act_window, save_image);
    
	    //Display other maps
		if(display == 7)
			cv::imshow(act_window, merged_points);

		if(display == 8)
			cv::imshow(act_window, zed_map_nn_resized);

		if(display == 9)
    		plot_maps(depth_confidence, 1, output_geo, -1, act_window, save_image);

	    cv::imshow("Original image", original_image);

		if(save_image){
			cv::imwrite("../images/original.jpeg", original_image);
			cv::imwrite("../images/normalized.jpeg", zed_map_nn_resized);
       		cv::imwrite("../images/merged.jpeg", merged_points);
	    }

		char key = cv::waitKey(10);
	    save_image = false;

		switch(key){
			case 's':
				save_image = true;

			case 'q':
				quit = true;

			case '1':
				display = 1;
			   cv::destroyWindow(act_window);
				act_window = "Monocular estimation";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);	   
				break;

			case '2':
				display = 2;
			   cv::destroyWindow(act_window);
				act_window = "Stereo Estimation";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '3':
				display = 3;
			   cv::destroyWindow(act_window);
				act_window = "Stereo Estimation N10";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '4':
				display = 4;
			   cv::destroyWindow(act_window);
				act_window = "Merged Estimation";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '5':
				display = 5;
			   cv::destroyWindow(act_window);
				act_window = "Merged Estimation2";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '6':
				display = 6;
			   cv::destroyWindow(act_window);
				act_window = "Weights influence";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '7':
				display = 7;
			   cv::destroyWindow(act_window);
				act_window = "Points 4 merge";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '8':
				display = 8;
			   cv::destroyWindow(act_window);
				act_window = "Zed normalized";
				cv::namedWindow(act_window, CV_WINDOW_NORMAL);
				break;

			case '9':
				display = 9;
			    
			    if (IMPORT_CONFIDENCE)
			    {
			    	cv::destroyWindow(act_window);
			    	act_window = "Confidence";
					cv::namedWindow(act_window, CV_WINDOW_NORMAL);
			    }

				break;

		}

	 //cv::imwrite("../images/ori.png", original_image);

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

        if(quit){
        	std::cout << '\n' << "Leaving..." << std::endl;
        	cv::destroyAllWindows();
        	return 0;
        }

      //  std::cout<<"FPS: "<< 1/duration <<'\n';

    }

    return 0;
}
