#include "manageObjectInputMap.h"

manageObjectInputMap::manageObjectInputMap(std::string typeOfMap, cv::Size desiredInputSize){

	this->setInputMapType(typeOfMap);
	this->setPath2SourceFolderInputMap();
	this->setInputMapFormat();
	this->setNumberFirstInputMap();
	this->setPath2InputMap();
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

void manageObjectInputMap::setPath2SourceFolderInputMap(){

	if(strcmp("depth", this->inputMapType.c_str() ) == 0){
		std::cout << "Insert path to folder with input depth map:" << std::endl;
		this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/mine/depth_maps/2/depths/gt/";
	}

	else if(strcmp("image", this->inputMapType.c_str()) == 0){
		std::cout << "Insert path to folder with input image:" << std::endl;
		this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/mine/depth_maps/2/images/left/";
	}

	else if(strcmp("confidence", this->inputMapType.c_str()) == 0){
		std::cout << "Insert path to folder with confidence map:" << std::endl;
		this->path2SourceFolderInputMap = "/home/diogo/Desktop/datasets/mine/depth_maps/2/depths/confidence/";
	}

	//std::cin.sync();
	//std::cin >> this->path2SourceFolderInputMap;


}

void manageObjectInputMap::setInputMapFormat(){

	std::cout << "Insert Image format:"<< std::endl;
	//std::cin.sync();
	//std::cin >> this->inputMapFormat;
	this->inputMapFormat = "20000_f";
}

void manageObjectInputMap::setNumberFirstInputMap(){

	std::string userInput;
	std::cin.sync();
	std::cout << "Insert number of first frame:"<< std::endl;
	//std::cin >>  userInput;
	//this->numberCurrentInputMap = stoi(userInput);
	this->numberCurrentInputMap = 2;

}

void manageObjectInputMap::readInputMap(){

	this->inputMap = cv::imread(this->path2InputMap, 1);

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

	if(strcmp("image", inputMapType.c_str()) == 0)
		cv::imshow("Input Image", this->getInputMap());

	else if(strcmp("depth", inputMapType.c_str()) == 0)
		cv::imshow("Input Depth Map", this->getInputMap());		

	else if(strcmp("confidence", inputMapType.c_str()) == 0)
		cv::imshow("Confidence Depth Map", this->getInputMap());		
}


void manageObjectInputMap::displayInputMapResized(){

	if(strcmp("image", inputMapType.c_str()) == 0)
		cv::imshow("Input Image", this->getInputMapResized());

	else if(strcmp("depth", inputMapType.c_str()) == 0)
		cv::imshow("Input Depth Map", this->getInputMapResized());		

	else if(strcmp("confidence", inputMapType.c_str()) == 0)
		cv::imshow("Confidence Depth Map", this->getInputMapResized());		
}

void manageObjectInputMap::setPath2InputMap(){

	this->path2InputMap = this->path2SourceFolderInputMap + this->inputMapFormat + std::to_string(this->numberCurrentInputMap) + ".png";
};

void manageObjectInputMap::updatePath2InputMap(){

	this->numberCurrentInputMap++;
	this->setPath2InputMap();

}

void manageObjectInputMap::setInputMapType(std::string inputMapType){

	this->inputMapType = inputMapType;

}

void manageObjectInputMap::setSizeInputMap(cv::Size desiredInputSize){


	this->desiredSizeInputMap.height = desiredInputSize.height;
	this->desiredSizeInputMap.width  = desiredInputSize.width;

}

void manageObjectInputMap::createInputMatrixResized(){

	if(strcmp("image", inputMapType.c_str()) == 0)
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC3);

	else if(strcmp("depth", inputMapType.c_str()) == 0)
		(this->inputMapResized).create( (this->desiredSizeInputMap).height, (this->desiredSizeInputMap).width, CV_32FC1);

}

void manageObjectInputMap::resizeInputMap(){

	cv::resize(this->inputMap, this->inputMapResized, this->desiredSizeInputMap);

}



