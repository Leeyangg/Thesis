#include "stereoAlgorithms.h"

stereoBMOpencv::stereoBMOpencv(){

	//this->stereoBM.create(16,15);
}

stereoBMOpencv::~stereoBMOpencv(){}

void stereoBMOpencv::computeDisparityMap(){

	this->stereoBM(this->leftImageGrayScale,this->rightImageGrayScale,this->disparityMap, CV_32FC1);
}

void stereoBMOpencv::computeAbsoluteDepthMap(){
	int heightMap = this->disparityMap.rows;
	int widthMap = this->disparityMap.cols;

	this->absoluteDepthMap.create(heightMap, widthMap, CV_32FC1);   

	for (int row = 0; row < heightMap; ++row)
	{
		for (int col = 0; col < widthMap; ++col)
		{	

			this->absoluteDepthMap.at<float>(row,col) = 10.0;
			this->disparityMap.at<float>(row,col) = std::abs(this->disparityMap.at<float>(row,col));
			this->absoluteDepthMap.at<float>(row,col) = this->scaleDepthMap/this->disparityMap.at<float>(row,col);
			
			if(this->absoluteDepthMap.at<float>(row,col) > 10.0)
				this->absoluteDepthMap.at<float>(row,col) = -99;	
		}
	}
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
