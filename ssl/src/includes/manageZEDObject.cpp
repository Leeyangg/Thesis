#include "manageZEDObject.h"

manageZEDObject::manageZEDObject(){
#ifdef COMPILE_ZED
	this->zedObject = new sl::zed::Camera(sl::zed::HD720); 	
	this->checkZEDStart();
	this->setMapWidth();
	this->setMapHeight();
   this->zedCVImage.create(this->mapHeight,this->mapWidth, CV_16UC3);
	this->zedCVMap.create(this->mapHeight,this->mapWidth, CV_32FC1);
	this->rightImage.create(this->mapHeight,this->mapWidth, this->zedCVImage.type());
	this->leftImage.create(this->mapHeight, this->mapWidth, this->zedCVImage.type());
	this->scaleToConvertMapToMeters = 0.001;
	this->setMaximumDepthDistance(10000);
	this->maximumDepth = this->zedObject->getDepthClampValue()*this->scaleToConvertMapToMeters;
	std::thread release(grabFrameZed, this);
	release.detach();

#else
	this->zedOpencv.open(0);

	this->checkZEDStart();
	this->setMapWidth();
	this->setMapHeight();
	this->rightImage.create(this->mapHeight,this->mapWidth, this->zedCVImage.type());
	this->leftImage.create(this->mapHeight, this->mapWidth, this->zedCVImage.type());
	this->zedCVMap.create(this->mapHeight,this->mapWidth, CV_32FC1);
	this->scaleToConvertMapToMeters = 700.897*0.120;
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

#else
	for (int i = 0; i < 10; ++i)
	{
		this->zedOpencv >> this->zedCVImage;
		this->zedOpencv >> this->zedCVImage;
	}

#endif
}

void manageZEDObject::grabFrame(){
#ifdef COMPILE_ZED
	this->zedObject->grab(sl::zed::RAW);
	sl::zed::slMat2cvMat(this->zedObject->retrieveImage(sl::zed::SIDE::LEFT)).copyTo(this->leftImage);
	sl::zed::slMat2cvMat(this->zedObject->retrieveImage(sl::zed::SIDE::RIGHT)).copyTo(this->rightImage);

#else

	this->zedOpencv >> this->zedCVImage;
	uchar* currentPointerToMemoryDestinationLeftImage;
	uchar* currentPointerToMemoryDestinationRightImage;
	uchar* currentPointerToMemorySource =  this->zedCVImage.ptr(0) ;

	for(int currentRowMatrixDepth = 0; currentRowMatrixDepth <  this->mapHeight; currentRowMatrixDepth++){
	
		currentPointerToMemoryDestinationRightImage = rightImage.ptr(currentRowMatrixDepth);
		currentPointerToMemoryDestinationLeftImage  = leftImage.ptr(currentRowMatrixDepth);

		memcpy(currentPointerToMemoryDestinationLeftImage, currentPointerToMemorySource,this->mapWidth*sizeof(uchar)*3);
		memcpy(currentPointerToMemoryDestinationRightImage, currentPointerToMemorySource +  this->mapWidth*sizeof(uchar)*3 , this->mapWidth*sizeof(uchar)*3);
		currentPointerToMemorySource = currentPointerToMemorySource + 2*this->mapWidth*sizeof(uchar)*3;

	}
	this->currentFrame++;
#endif

}

cv::Mat manageZEDObject::getImage(){	
#ifdef COMPILE_ZED
		sl::zed::slMat2cvMat(this->zedObject->retrieveImage(sl::zed::SIDE::LEFT)).copyTo(this->zedCVImage);
		return(this->zedCVImage);
#else
		return(this->leftImage);
#endif

}

cv::Mat manageZEDObject::getRightImage(bool save){	
	if(save){
		std::string path;
		path = "/media/diogo/My Passport/indoorsDepths/right/" + std::to_string(this->currentFrame) + ".png";
		cv::imwrite(path, this->rightImage);
	}
	return(this->rightImage);
}

cv::Mat manageZEDObject::getLeftImage(bool save){	
	if(save){
		std::string path;
		path = "/media/diogo/My Passport/indoorsDepths/left/" + std::to_string(this->currentFrame) + ".png";
		cv::imwrite(path, this->leftImage);
	}

	return(this->leftImage);
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
		cv::convertScaleAbs(this->zedCVMap, this->zedCVMap, 255*this->scaleToConvertMapToMeters/this->maximumDepth);
#else

#endif


//map between [0,255]
	return(this->zedCVMap);

}

void manageZEDObject::setMapWidth(){
#ifdef COMPILE_ZED
	this->mapWidth = this->zedObject->getImageSize().width;

#else
	this->mapWidth = this->zedCVImage.cols/2;
#endif
}

void manageZEDObject::setMapHeight(){
#ifdef COMPILE_ZED
	this->mapHeight = this->zedObject->getImageSize().height;
#else
	this->mapHeight = this->zedCVImage.rows;
#endif
}

void grabFrameZed(manageZEDObject* zedCamObject){

	for(;;){
		zedCamObject->grabFrame();
	}

}
