#include "manageObjectInputMap.h"

manageObjectInputMap::manageObjectInputMap(std::string typeOfMap, cv::Size desiredInputSize){

	this->setInputMapType(typeOfMap);
	this->setPath2SourceInputMap();
	this->setInputMapFormat();
	this->setNumberFirstInputMap();
	this->setPath2InputMap();
	this->setSizeInputMap(desiredInputSize);
	this->createInputMatrixResized();


}

 manageObjectInputMap::manageObjectInputMap(std::string typeOfMap, cv::Size desiredInputSize, manageZEDObject* zedCamObject){

	this->setInputMapType(typeOfMap);
    this->zedCamObject = zedCamObject;
	this->setSizeInputMap(desiredInputSize);
	this->createInputMatrixResized();

}

manageObjectInputMap::~manageObjectInputMap(){}

cv::Mat manageObjectInputMap::getInputMap(){

	return this->inputMap;

}

cv::Mat manageObjectInputMap::getInputMapResized(){

	return this->inputMapResized;

}

void manageObjectInputMap::setPath2SourceInputMap(){

	if(this->inputMapType == DEPTH_MAP_){
		std::cout << "Insert path to folder with input depth map:" << std::endl;
		this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/nyu1_2/train_gt/labels/";
		//this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/mine/depth_maps/5/depths/gt/";
	}

	else if(this->inputMapType == IMAGE_MAP_){
		std::cout << "Insert path to folder with input image:" << std::endl;
		this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/nyu1_2/train/labels/";
		//this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/mine/depth_maps/5/images/left/";
	}

	else if(this->inputMapType == CONFIDENCE_MAP_){
		std::cout << "Insert path to folder with confidence map:" << std::endl;
		this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/mine/depth_maps/5/depths/confidence/";
	}

	//std::cin.sync();
	//std::cin >> this->path2SourceFolderInputMap;


}

void manageObjectInputMap::setInputMapFormat(){

	std::cout << "Insert Image format:"<< std::endl;
	//std::cin.sync();
	//std::cin >> this->inputMapFormat;
	this->inputMapFormat = "";
}

void manageObjectInputMap::setNumberFirstInputMap(){

	std::cout << "Insert number of first frame:"<< std::endl;
	//std::cin >>  userInput;
	//this->numberCurrentInputMap = stoi(userInput);
	this->numberCurrentInputMap = 2;

}

bool manageObjectInputMap::testInputMapExists(){

	if( !(this->inputMap.data) )
		this->inputMapExists = false;

	else
		this->inputMapExists = true;

	return(this->inputMapExists);
}

std::string manageObjectInputMap::getPath2InputMap(){

	return(path2InputMap);

}

void manageObjectInputMap::displayInputMap(){

	if(this->inputMapType == IMAGE_MAP_ || this->inputMapType == ZED_CAM_MAP_)
		cv::imshow("Input Image", this->getInputMap());

	else if(this->inputMapType == DEPTH_MAP_ || this->inputMapType == ZED_DEPTH_MAP_ )
		cv::imshow("Input Depth Map", this->getInputMap());		

	else if(this->inputMapType == CONFIDENCE_MAP_ || this->inputMapType == ZED_CONFIDENCE_MAP_ )
		cv::imshow("Confidence Depth Map", this->getInputMap());	

	
}


void manageObjectInputMap::displayInputMapResized(){

	if(this->inputMapType == IMAGE_MAP_ || this->inputMapType == ZED_CAM_MAP_ )
		cv::imshow("Input Image", this->getInputMapResized());

	else if(this->inputMapType == DEPTH_MAP_ || this->inputMapType == ZED_DEPTH_MAP_  )
		cv::imshow("Input Depth Map", this->getInputMapResized());		

	else if(this->inputMapType == CONFIDENCE_MAP_ || this->inputMapType == ZED_CONFIDENCE_MAP_ )
		cv::imshow("Confidence Depth Map", this->getInputMapResized());		
	
}

void manageObjectInputMap::setPath2InputMap(){

	this->path2InputMap = this->path2SourceFolderInputMap + this->inputMapFormat + std::to_string(this->numberCurrentInputMap) + ".png";

}

void manageObjectInputMap::updateInputMap(){


	if( (this->inputMapType != ZED_CAM_MAP_) && (this->inputMapType != ZED_CONFIDENCE_MAP_) && (this->inputMapType != ZED_DEPTH_MAP_) ){	
		this->numberCurrentInputMap++;
		this->setPath2InputMap();
	}

}

void manageObjectInputMap::setInputMapType(std::string inputMapType){

	if(strcmp("image", inputMapType.c_str()) == 0)
		this->inputMapType = IMAGE_MAP_;

	else if(strcmp("depth", inputMapType.c_str()) == 0)
		this->inputMapType = DEPTH_MAP_;		

	else if(strcmp("confidence", inputMapType.c_str()) == 0)
		this->inputMapType = CONFIDENCE_MAP_;	

	else if(strcmp("zed_confidence", inputMapType.c_str()) == 0)
		this->inputMapType = ZED_CONFIDENCE_MAP_;	

	else if(strcmp("zed", inputMapType.c_str()) == 0)
		this->inputMapType = ZED_CAM_MAP_;	

	else if(strcmp("zed_depth", inputMapType.c_str()) == 0)
		this->inputMapType = ZED_DEPTH_MAP_;	


}

void manageObjectInputMap::setSizeInputMap(cv::Size desiredInputSize){


	this->desiredSizeInputMap.height = desiredInputSize.height;
	this->desiredSizeInputMap.width  = desiredInputSize.width;

}

void manageObjectInputMap::createInputMatrixResized(){

	if(this->inputMapType == IMAGE_MAP_ || this->inputMapType == ZED_CAM_MAP_)
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC3);

	else 
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC1);

}

void manageObjectInputMap::resizeInputMap(){

	cv::resize(this->inputMap, this->inputMapResized, this->desiredSizeInputMap);

}

void manageObjectInputMap::readInputMap(){

	if(this->inputMapType == ZED_CAM_MAP_)
		this->zedCamObject->getImage().copyTo(this->inputMap);

	else if(this->inputMapType == ZED_CONFIDENCE_MAP_)
		this->zedCamObject->getConfidenceMap().copyTo(this->inputMap);	

	else if(this->inputMapType == ZED_DEPTH_MAP_){
		this->zedCamObject->getDepthMap().copyTo(this->inputMap);	
}

	else if(this->inputMapType == IMAGE_MAP_)
		this->inputMap = cv::imread(this->path2InputMap, 1 ); 

	else{
			this->inputMap = cv::imread(this->path2InputMap, 0 ); 
		}
}