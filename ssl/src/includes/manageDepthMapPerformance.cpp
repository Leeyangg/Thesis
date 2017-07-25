#include "manageDepthMapPerformance.h"

extern  int thresholdConfidenceJSONFile;
extern  bool zedSourceSdkJSONFile;
extern  bool janivanecky;

manageDepthMapPerformance::manageDepthMapPerformance(){

	this->thresholdError = 0.0 ;
	this->absoluteRelativeError = 0.0;
	this->squaredRelativeError = 0.0;
	this->linearRMSE = 0.0;
	this->logRMSE = 0.0;
	this->scaleInvariantError = 0.0;
	this->thresholdErrorThreshold = pow(1.25,2);
	this->scaleDepthMap = 1.0;
	this->scaleGroundTruth = 1.0;

}


manageDepthMapPerformance::~manageDepthMapPerformance(){}

void manageDepthMapPerformance::setDepthMapGroundTruth(cv::Mat groundTruthMap){

	this->groundTruthMap = groundTruthMap;
	this->groundTruthMap.convertTo(this->groundTruthMap, CV_32FC1);

}

void manageDepthMapPerformance::resetErrors(){

	this->thresholdError = 0.0;
	this->absoluteRelativeError = 0.0;
	this->squaredRelativeError = 0.0;
	this->linearRMSE = 0.0;
	this->logRMSE = 0.0;
	this->scaleInvariantError = 0.0;
	this->scaleInvariantErrorStruct.partial1 = 0.0;
	this->scaleInvariantErrorStruct.partial2.clear();
}

void manageDepthMapPerformance::setDepthMapEstimation(cv::Mat estimationMap){

	this->estimationMap = estimationMap;
	this->estimationMap.convertTo(this->estimationMap, CV_32FC1);

}

void manageDepthMapPerformance::setScaleDepthMap(float scale){
	this->scaleDepthMap = scale;
}

void manageDepthMapPerformance::setScaleGroundTruth(float scale){
	this->scaleGroundTruth = scale;
}

void manageDepthMapPerformance::computePerformance(cv::Mat confidenceMap){

	confidenceMap.convertTo(confidenceMap, CV_32FC1);

	int rowsInputMap = this->groundTruthMap.rows;
	int colsInputMap = this->groundTruthMap.cols;
	int numberValidPixels = 0;
	
	this->resetErrors();

	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{
			if(this->estimationMap.at<float>(currentRow, currentCol) >= 0.0 && confidenceMap.at<float>(currentRow, currentCol) >= thresholdConfidenceJSONFile){			

				if(!zedSourceSdkJSONFile){
					this->currentPixelGroundTruth = this->groundTruthMap.at<float>(currentRow, currentCol)*(19.5/255.0*(-1.0))+20.0;
				}
				else
					this->currentPixelGroundTruth = this->groundTruthMap.at<float>(currentRow, currentCol)*this->scaleGroundTruth;

	    		this->currentPixelPrediction  = this->estimationMap.at<float>(currentRow, currentCol)*(19.5/255.0*(-1.0))+20.0;

				this->thresholdError          = this->thresholdError + this->computeThresholdError();
				this->absoluteRelativeError   = this->absoluteRelativeError +  this->computeAbsoluteRelativeError();
				this->squaredRelativeError    = this->squaredRelativeError + this->computeSquaredRelativeError();
				this->linearRMSE              = this->linearRMSE + this->computeLinearRMSE();
				this->logRMSE                 = this->logRMSE +  this->computeLogRMSE();
				this->computePartialsScaleInvariantError();
				numberValidPixels++;
			}
		}
	}

	if(numberValidPixels > 0){
		this->thresholdError          = this->thresholdError/(numberValidPixels);
		this->absoluteRelativeError   = this->absoluteRelativeError/(numberValidPixels);
		this->squaredRelativeError    = this->squaredRelativeError/(numberValidPixels);
		this->linearRMSE              = sqrt(this->linearRMSE/(numberValidPixels));
		this->logRMSE                 = sqrt(this->logRMSE/(numberValidPixels));
		this->scaleInvariantErrorStruct.partial1  = this->scaleInvariantErrorStruct.partial1/numberValidPixels;
		this->scaleInvariantError     =  0.5*this->computeScaleInvariantError()/numberValidPixels;
	}
	else{
		this->thresholdError          = -99;
		this->absoluteRelativeError   = -99;
		this->squaredRelativeError    = -99;
		this->linearRMSE              = -99;
		this->logRMSE                 = -99;
		this->scaleInvariantError     = -99;
	}

}


void manageDepthMapPerformance::computePerformance(){

	int rowsInputMap = this->groundTruthMap.rows;
	int colsInputMap = this->groundTruthMap.cols;
	int numberValidPixels = 0;
	cv::Mat errorMat(rowsInputMap,colsInputMap, CV_32FC1);
	this->resetErrors();
	double min, max;

	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{
			if(this->estimationMap.at<float>(currentRow, currentCol) >= 0.0){			

				if(!zedSourceSdkJSONFile)
					this->currentPixelGroundTruth = this->groundTruthMap.at<float>(currentRow, currentCol)*(19.5/255.0*(-1.0))+20.0;
				else
					this->currentPixelGroundTruth = this->groundTruthMap.at<float>(currentRow, currentCol)*this->scaleGroundTruth;
				
				if(this->scaleDepthMap == 1.0){
					this->currentPixelPrediction  = this->estimationMap.at<float>(currentRow, currentCol);
				}

				else if(this->scaleDepthMap == 99.5){
					this->currentPixelPrediction  = this->estimationMap.at<float>(currentRow, currentCol)*(19.5/255.0*(-1.0))+20.0;
				}

			    else{
			    	if(!janivanecky)
			    		this->currentPixelPrediction  = this->estimationMap.at<float>(currentRow, currentCol)*(19.5*(-1.0))+20.0;
			    	else{
			    		this->currentPixelPrediction = 19.5*this->estimationMap.at<float>(currentRow, currentCol) + 0.5;
			    	}
			    }

			     errorMat.at<float>(currentRow, currentCol) = this->computeLogRMSE();

				this->thresholdError          = this->thresholdError + this->computeThresholdError();
				this->absoluteRelativeError   = this->absoluteRelativeError +  this->computeAbsoluteRelativeError();
				this->squaredRelativeError    = this->squaredRelativeError + this->computeSquaredRelativeError();
				this->linearRMSE              = this->linearRMSE + this->computeLinearRMSE();
				this->logRMSE                 = this->logRMSE +  this->computeLogRMSE();
				this->computePartialsScaleInvariantError();
				numberValidPixels++;
			}
		}
	}

	cv::minMaxLoc(errorMat, &min, &max);
	errorMat = errorMat/max;
	if(this->scaleDepthMap == 1.0)
		cv::imshow("log", errorMat);

	if(numberValidPixels > 0){
		this->thresholdError          = this->thresholdError/(numberValidPixels);
		this->absoluteRelativeError   = this->absoluteRelativeError/(numberValidPixels);
		this->squaredRelativeError    = this->squaredRelativeError/(numberValidPixels);
		this->linearRMSE              = sqrt(this->linearRMSE/(numberValidPixels));
		this->logRMSE                 = sqrt(this->logRMSE/(numberValidPixels));
		this->scaleInvariantErrorStruct.partial1  = this->scaleInvariantErrorStruct.partial1/numberValidPixels;
		this->scaleInvariantError     =  0.5*this->computeScaleInvariantError()/numberValidPixels;
	}

	else{
		this->thresholdError          = -99;
		this->absoluteRelativeError   = -99;
		this->squaredRelativeError    = -99;
		this->linearRMSE              = -99;
		this->logRMSE                 = -99;
		this->scaleInvariantError     = -99;
	}


}


float manageDepthMapPerformance::computeThresholdError(){

	if(  (this->currentPixelGroundTruth/this->currentPixelPrediction < this->thresholdErrorThreshold) && (this->currentPixelPrediction/this->currentPixelGroundTruth < this->thresholdErrorThreshold) ){
		return 1.0;
	}

	else
		return 0.0;

}

float manageDepthMapPerformance::computeAbsoluteRelativeError(){

	return( std::abs( (this->currentPixelGroundTruth - this->currentPixelPrediction) ) / this->currentPixelGroundTruth );
	
}

float manageDepthMapPerformance::computeSquaredRelativeError(){

	return( pow( this->currentPixelGroundTruth - this->currentPixelPrediction ,2)/ this->currentPixelGroundTruth );
 	
}

float manageDepthMapPerformance::computeLinearRMSE(){

	return( pow(this->currentPixelPrediction - this->currentPixelGroundTruth,2) );
	
}

double manageDepthMapPerformance::computeLogRMSE(){
	double inter;
	inter = log((double)this->currentPixelPrediction) -  log((double)this->currentPixelGroundTruth);
	return( pow(inter,2) );
	
}

void manageDepthMapPerformance::computePartialsScaleInvariantError(){

	this->scaleInvariantErrorStruct.partial1 = this->scaleInvariantErrorStruct.partial1 + (log(this->currentPixelGroundTruth) - log(this->currentPixelPrediction)) ;
	this->scaleInvariantErrorStruct.partial2.push_back(log(this->currentPixelPrediction) - log(this->currentPixelGroundTruth));
	
}

float manageDepthMapPerformance::computeScaleInvariantError(){

	float finalPartials = 0.0;

	for (auto scaleInv : this->scaleInvariantErrorStruct.partial2 ){
	
		finalPartials = finalPartials + pow((scaleInv + this->scaleInvariantErrorStruct.partial1),2);

	}

	return(finalPartials);
	
}

float manageDepthMapPerformance::getThresholdError(){

	return(this->thresholdError );

}

float manageDepthMapPerformance::getAbsoluteRelativeError(){

	return(this->absoluteRelativeError);

}


float manageDepthMapPerformance::getSquaredRelativeError(){

	return(this->squaredRelativeError);

}

float manageDepthMapPerformance::getLinearRMSE(){

	return(this->linearRMSE);

}
double manageDepthMapPerformance::getLogRMSE(){

	return(this->logRMSE);

}

float manageDepthMapPerformance::getScaleInvariantError(){

	return(this->scaleInvariantError);

}

void manageDepthMapPerformance::setThresholdErrorThreshold(float threshold){

	this->thresholdErrorThreshold = threshold;

}
