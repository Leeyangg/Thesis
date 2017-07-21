#include "manageObjectInputMap.h"

extern  std::string pathLeftImageJSONFile;
extern  std::string pathRightImageJSONFile;
extern  std::string formatNameImagesJSONFile;
extern  std::string sourceGt;
extern  std::string sourceCheapDepth;
extern  std::string confidencePathJSONFile;
extern  int numberFirstFrameJSONFile;


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

cv::Mat manageObjectInputMap::getRightMap(){

	return this->rightMap;

}

cv::Mat manageObjectInputMap::getRightMapResized(){

	return this->rightMapResized;

}

void manageObjectInputMap::setPath2SourceInputMap(){

	if(this->inputMapType == DEPTH_MAP_){
		this->path2SourceFolderInputMap = sourceGt;
	}

	else if(this->inputMapType == CHEAP_DEPTH_){
		this->path2SourceFolderInputMap = sourceCheapDepth;
	}

	else if(this->inputMapType == IMAGE_MAP_){
		this->path2SourceFolderInputMap = pathLeftImageJSONFile;
	}

	else if(this->inputMapType == CONFIDENCE_MAP_){
		this->path2SourceFolderInputMap = confidencePathJSONFile;
	}

	else if(this->inputMapType == STEREO_IMAGE_){
		this->path2SourceFolderInputMap = pathLeftImageJSONFile;
		this->path2SourceFolderRightMap = pathRightImageJSONFile;
	}



}

void manageObjectInputMap::setInputMapFormat(){
	this->inputMapFormat = formatNameImagesJSONFile;
}

void manageObjectInputMap::setNumberFirstInputMap(){
	this->numberCurrentInputMap = numberFirstFrameJSONFile;
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

	else if(this->inputMapType == CHEAP_DEPTH_ )
		cv::imshow("Cheap Depth Map", this->getInputMap());		

	
}


void manageObjectInputMap::displayInputMapResized(){

	if(this->inputMapType == IMAGE_MAP_ || this->inputMapType == ZED_CAM_MAP_ )
		cv::imshow("Input Image", this->getInputMapResized());

	else if(this->inputMapType == DEPTH_MAP_ || this->inputMapType == ZED_DEPTH_MAP_  )
		cv::imshow("Input Depth Map", this->getInputMapResized());		

	else if(this->inputMapType == CONFIDENCE_MAP_ || this->inputMapType == ZED_CONFIDENCE_MAP_ )
		cv::imshow("Confidence Depth Map", this->getInputMapResized());		

	else if(this->inputMapType == CHEAP_DEPTH_ )
		cv::imshow("Cheap Depth Map", this->getInputMapResized());	
	
}

void manageObjectInputMap::setPath2InputMap(){

	this->path2InputMap = this->path2SourceFolderInputMap + this->inputMapFormat + std::to_string(this->numberCurrentInputMap) + ".png";

	if(this->inputMapType == STEREO_IMAGE_)
		this->path2RightMap = this->path2SourceFolderRightMap + this->inputMapFormat + std::to_string(this->numberCurrentInputMap) + ".png";


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

	else if(strcmp("stereo_pair", inputMapType.c_str()) == 0)
		this->inputMapType = STEREO_IMAGE_;	

	else if(strcmp("cheap_depth", inputMapType.c_str()) == 0)
		this->inputMapType =  CHEAP_DEPTH_ ;	

}

void manageObjectInputMap::setSizeInputMap(cv::Size desiredInputSize){


	this->desiredSizeInputMap.height = desiredInputSize.height;
	this->desiredSizeInputMap.width  = desiredInputSize.width;

}

void manageObjectInputMap::createInputMatrixResized(){

	if(this->inputMapType == IMAGE_MAP_ || this->inputMapType == ZED_CAM_MAP_)
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC3);

	else if(this->inputMapType == STEREO_IMAGE_){
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC3);
		(this->rightMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC3);
	}

	else 
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC1);

}

void manageObjectInputMap::resizeInputMap(){

	cv::resize(this->inputMap, this->inputMapResized, this->desiredSizeInputMap);
	
	if(this->inputMapType == STEREO_IMAGE_)
		cv::resize(this->rightMap, this->rightMapResized, this->desiredSizeInputMap);

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

	else if(this->inputMapType == STEREO_IMAGE_){
		this->inputMap = cv::imread(this->path2InputMap, 1 ); 
		this->rightMap = cv::imread(this->path2RightMap, 1 );
	}

	else{
			this->inputMap = cv::imread(this->path2InputMap, 0 ); 
		}
}