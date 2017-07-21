#include "stereoAlgorithms.h"

stereoBMOpencv::stereoBMOpencv(){
	this->stereoBM.init(0,0,9);
	int numberOfDisparities = 80;
	this->stereoSGBM.preFilterCap =16;
	this->stereoSGBM.SADWindowSize = 3;
	this->stereoSGBM.P1 = 8*3*this->stereoSGBM.SADWindowSize*this->stereoSGBM.SADWindowSize;
	this->stereoSGBM.P2 = 32*3*this->stereoSGBM.SADWindowSize*this->stereoSGBM.SADWindowSize;
	this->stereoSGBM.minDisparity = 1;
	this->stereoSGBM.numberOfDisparities = numberOfDisparities; 
	this->stereoSGBM.uniquenessRatio = 1;
	this->stereoSGBM.speckleWindowSize = 0;
	this->stereoSGBM.speckleRange = 2;
	this->stereoSGBM.disp12MaxDiff = -1;
	this->stereoSGBM.fullDP = true;

}

stereoBMOpencv::~stereoBMOpencv(){}

void stereoBMOpencv::computeDisparityMap(){

	cv::Mat intDisparitites;
	//this->stereoBM(this->leftImageGrayScale,this->rightImageGrayScale,this->disparityMap, CV_32FC1);
	 intDisparitites.create(this->leftImageGrayScale.rows, this->leftImageGrayScale.cols, CV_16SC1);
	this->stereoSGBM(this->leftImageGrayScale,this->rightImageGrayScale,   intDisparitites);
	 intDisparitites =  intDisparitites/16.0;
	 intDisparitites.convertTo( intDisparitites, CV_32FC1);
	cv::resize(intDisparitites,this->disparityMap,this->resolution);

}

void stereoBMOpencv::computeAbsoluteDepthMap(){

	int heightMap = this->disparityMap.rows;
	int widthMap = this->disparityMap.cols;

	this->pointsForSSL.create(heightMap, widthMap, CV_32FC1);
	this->absoluteDepthMap.create(heightMap, widthMap, CV_32FC1);   

	this->availablePoints = 0;

	for (int row = 0; row < heightMap; ++row)
	{
		for (int col = 0; col < widthMap; ++col)
		{	

			if(this->disparityMap.at<float>(row,col) < 0.0)
				this->disparityMap.at<float>(row,col) = this->disparityMap.at<float>(row,col)*-1;

			if(this->disparityMap.at<float>(row,col)!= 0)
				this->absoluteDepthMap.at<float>(row,col) = this->scaleDepthMap/this->disparityMap.at<float>(row,col);

			else
				this->absoluteDepthMap.at<float>(row,col) = 20.0;
			
			if( (this->absoluteDepthMap.at<float>(row,col) > 0.0) && (this->absoluteDepthMap.at<float>(row,col) < 20.0)){
				this->pointsForSSL.at<float>(row,col) = 1.0;
				this->availablePoints++;
			}		

			else 
				this->absoluteDepthMap.at<float>(row,col) = -99.0;

		}
	}
	
}

int stereoBMOpencv::getAvailablePoints(){
	return(this->availablePoints);
}

cv::Mat stereoBMOpencv::getPointsForSSL(){
	cv::Mat pointsResized;
	cv::resize(this->pointsForSSL, pointsResized, this->resolution);
	return(pointsResized);

}

void stereoBMOpencv::setLeftImage(cv::Mat image){
	image.copyTo(this->leftImage);	
	cv::cvtColor(this->leftImage, this->leftImageGrayScale, CV_BGR2GRAY);
}

void stereoBMOpencv::setRightImage(cv::Mat image){
	image.copyTo(this->rightImage);
	cv::cvtColor(this->rightImage, this->rightImageGrayScale, CV_BGR2GRAY);	
}

cv::Mat stereoBMOpencv::getDisparityMap(){
	return(this->disparityMap);
}

cv::Mat stereoBMOpencv::getAbsoluteDepthMap(){
	return(this->absoluteDepthMap);
}

void stereoBMOpencv::setScaleDepthMap(float scale){
	this->scaleDepthMap = scale;
}

void stereoBMOpencv::setResolution(cv::Size resolution){

	this->resolution.width = resolution.width;
	this->resolution.height = resolution.height;

}

cv::Mat stereoBMOpencv::getDisparityMapResized(){

	cv::resize(this->disparityMap, this->disparityMapResized, this->resolution);
	return(this->disparityMapResized);
}

cv::Mat stereoBMOpencv::getAbsoluteDepthMapResized(){

	cv::resize(this->absoluteDepthMap, this->absoluteDepthMapResized, this->resolution);
	return(this->absoluteDepthMapResized);
}
