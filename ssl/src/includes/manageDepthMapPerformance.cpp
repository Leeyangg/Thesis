#include "manageDepthMapPerformance.h"


manageDepthMapPerformance::manageDepthMapPerformance(){

	this->thresholdError = 0.0 ;
	this->absoluteRelativeError = 0.0;
	this->squaredRelativeError = 0.0;
	this->linearRMSE = 0.0;
	this->logRMSE = 0.0;
	this->scaleInvariantError = 0.0;
	this->cumulativeError = 0.0;
	this->meanError = 0.0;
	this->counterMeanError = 1;
	this->thresholdErrorThreshold = 1.25;
	this->scaleDepthMap = 1.0;
	this->scaleGroundTruth = 1.0;

}


manageDepthMapPerformance::~manageDepthMapPerformance(){}

void manageDepthMapPerformance::setDepthMapGroundTruth(cv::Mat groundTruthMap){

	this->groundTruthMap = groundTruthMap;
	this->groundTruthMap.convertTo(this->groundTruthMap, CV_32FC1);

}

void manageDepthMapPerformance::resetErrors(){

	this->absoluteRelativeError = 0.0;
	this->squaredRelativeError = 0.0;
	this->linearRMSE = 0.0;
	this->logRMSE = 0.0;
	this->scaleInvariantError = 0.0;
}

void manageDepthMapPerformance::computeMeanError(float partialError){
	this->computeCumulativeError(partialError);
	this->meanError = this->cumulativeError/float(this->counterMeanError);
	this->counterMeanError++;
}

void manageDepthMapPerformance::computeCumulativeError(float partialError){
	this->cumulativeError = this->cumulativeError + partialError;
}


float manageDepthMapPerformance::getMeanError(){
	return(this->meanError);
}

float manageDepthMapPerformance::getCumulativeError(){
	return(this->cumulativeError);
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



void manageDepthMapPerformance::computePerformance(){

	int rowsInputMap = this->groundTruthMap.rows;
	int colsInputMap = this->groundTruthMap.cols;
	int numberValidPixels = 0;
	
	this->resetErrors();

	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{
			if(this->estimationMap.at<float>(currentRow, currentCol) >= 0.0){			

				this->currentPixelGroundTruth = this->groundTruthMap.at<float>(currentRow, currentCol)*this->scaleGroundTruth;
				this->currentPixelPrediction  = this->estimationMap.at<float>(currentRow, currentCol)*this->scaleDepthMap;

				this->thresholdError          = this->thresholdError + this->computeThresholdError();
				this->absoluteRelativeError   = this->absoluteRelativeError +  this->computeAbsoluteRelativeError();
				this->squaredRelativeError    = this->squaredRelativeError + this->computeSquaredRelativeError();
				this->linearRMSE              = this->linearRMSE + this->computeLinearRMSE();
				this->logRMSE                 = this->logRMSE +  this->computeLogRMSE();
				this->computeScaleInvariantError();
				numberValidPixels++;
			}
		}
	}

	this->thresholdError          = this->thresholdError/(numberValidPixels);
	this->absoluteRelativeError   = this->absoluteRelativeError/(numberValidPixels);
	this->squaredRelativeError    = this->squaredRelativeError/(numberValidPixels);
	this->linearRMSE              = sqrt(this->linearRMSE/(numberValidPixels));
	this->logRMSE                 = sqrt(this->logRMSE/(numberValidPixels));
	this->scaleInvariantError     = (this->scaleInvariantErrorStruct.partial1)/(numberValidPixels) - 0.5*( pow(this->scaleInvariantErrorStruct.partial2,2)) / (pow(numberValidPixels,2));

}


float manageDepthMapPerformance::computeThresholdError(){

	if(  (this->currentPixelGroundTruth/this->currentPixelPrediction < this->thresholdErrorThreshold) && (this->currentPixelPrediction/this->currentPixelGroundTruth < this->thresholdErrorThreshold) ){
		return 1.0;
	}

	else
		return 0.0;

}

float manageDepthMapPerformance::computeAbsoluteRelativeError(){

	return( abs((this->currentPixelGroundTruth - this->currentPixelPrediction)) / this->currentPixelGroundTruth );
	
}

float manageDepthMapPerformance::computeSquaredRelativeError(){

	return( pow( this->currentPixelGroundTruth - this->currentPixelPrediction ,2)/ this->currentPixelGroundTruth );
 	
}

float manageDepthMapPerformance::computeLinearRMSE(){

	return( pow(this->currentPixelPrediction - this->currentPixelGroundTruth,2) );
	
}

float manageDepthMapPerformance::computeLogRMSE(){

	return( pow(log(this->currentPixelPrediction) - log(this->currentPixelGroundTruth),2) );
	
}

void manageDepthMapPerformance::computeScaleInvariantError(){

	this->scaleInvariantErrorStruct.di = log(this->currentPixelPrediction) - log(this->currentPixelGroundTruth);
	this->scaleInvariantErrorStruct.partial1 = this->scaleInvariantErrorStruct.partial1 + pow(this->scaleInvariantErrorStruct.di,2);
	this->scaleInvariantErrorStruct.partial2 = this->scaleInvariantErrorStruct.partial2 + this->scaleInvariantErrorStruct.di;
	
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
float manageDepthMapPerformance::getLogRMSE(){

	return(this->logRMSE);

}

float manageDepthMapPerformance::getScaleInvariantError(){

	return(this->scaleInvariantError);

}

void manageDepthMapPerformance::setThresholdErrorThreshold(float threshold){

	this->thresholdErrorThreshold = threshold;

}
