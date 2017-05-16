#include "manageZEDObject.h"

manageZEDObject::manageZEDObject(){
#ifdef COMPILE_ZED
	this->zedObject = new sl::zed::Camera(sl::zed::HD720); 	
	this->checkZEDStart();
	this->setMapWidth();
	this->setMapHeight();
    this->zedCVImage.create(this->mapHeight,this->mapWidth, CV_16UC3);
	this->zedCVMap.create(this->mapHeight,this->mapWidth, CV_32FC1);
	this->scaleToConvertMapToMeters = 0.001;
	this->setMaximumDepthDistance(10000);
	this->maximumDepth = this->zedObject->getDepthClampValue()*this->scaleToConvertMapToMeters;
#endif
}

manageZEDObject::~manageZEDObject(){}

void manageZEDObject::setMaximumDepthDistance(float maximumDepth){

#ifdef COMPILE_ZED
	this->zedObject->setDepthClampValue(maximumDepth);
#endif

}

void manageZEDObject::checkZEDStart(){
#ifdef COMPILE_ZED

	sl::zed::ERRCODE err = this->zedObject->init(sl::zed::MODE::PERFORMANCE, 0,true,false,false);

	if (strcmp(sl::zed::errcode2str(err).c_str(), "SUCCESS") != 0) {
		std::cout <<" ZED not setup \n Leaving... " << std::endl;
		exit(EXIT_FAILURE);
    }

	else
		std::cout <<"ZED started " << std::endl;

#endif
}

void manageZEDObject::grabFrame(){
#ifdef COMPILE_ZED
	this->zedObject->grab(sl::zed::RAW);
#endif

}

cv::Mat manageZEDObject::getImage(){	
#ifdef COMPILE_ZED
		sl::zed::slMat2cvMat(this->zedObject->retrieveImage(sl::zed::SIDE::LEFT)).copyTo(this->zedCVImage);
#endif

	return(this->zedCVImage);

}

cv::Mat manageZEDObject::getConfidenceMap(){
#ifdef COMPILE_ZED
		sl::zed::slMat2cvMat(this->zedObject->retrieveMeasure(sl::zed::MEASURE::CONFIDENCE)).copyTo(this->zedCVMap);
#endif

	return(this->zedCVMap);

}

cv::Mat manageZEDObject::getDepthMap(){
#ifdef COMPILE_ZED
		sl::zed::slMat2cvMat(this->zedObject->retrieveMeasure(sl::zed::MEASURE::DEPTH)).copyTo(this->zedCVMap);
#endif

		cv::convertScaleAbs(this->zedCVMap, this->zedCVMap, 255*this->scaleToConvertMapToMeters/this->maximumDepth);
//map between [0,255]
	return(this->zedCVMap);

}

void manageZEDObject::setMapWidth(){
#ifdef COMPILE_ZED
	this->mapWidth = this->zedObject->getImageSize().width;
#endif
}

void manageZEDObject::setMapHeight(){
#ifdef COMPILE_ZED
	this->mapHeight = this->zedObject->getImageSize().height;
#endif
}
