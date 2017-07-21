#include "confidenceMergeAlgorithm.h"
extern bool janivanecky;

confidenceMergeAlgorithm::confidenceMergeAlgorithm(){}

confidenceMergeAlgorithm::~confidenceMergeAlgorithm(){}

void confidenceMergeAlgorithm::merge(){
	
	double min, max;
    this->stereoDepthMap.convertTo(this->stereoDepthMap, CV_32FC1);
	this->monoDepthMap.convertTo(this->monoDepthMap, CV_32FC1);
	this->finalDepthMap.create(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);

	float averageDepthMono = 0.0;
	cv::Mat mergedDepthMap(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);
	cv::Mat mergedDepthMap2(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);
	int rowsInputMap = this->monoDepthMap.rows;
	int colsInputMap = this->monoDepthMap.cols;
	cv::Point_<int> currentPixel;

	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{

			if(stereoOpenCVJSONFile)
				this->stereoDepthMap.at<float>(currentRow, currentCol) = this->stereoDepthMap.at<float>(currentRow, currentCol)*this->scaleStereoDepthMap;

			else
				this->stereoDepthMap.at<float>(currentRow, currentCol) = this->stereoDepthMap.at<float>(currentRow, currentCol)*(19.5/255.0*(-1.0))+20.0;

	    	if(!janivanecky)
	    		this->monoDepthMap.at<float>(currentRow, currentCol) =  this->monoDepthMap.at<float>(currentRow, currentCol)*(19.50*(-1.0))+20.0;

	    	else{
	    		this->monoDepthMap.at<float>(currentRow, currentCol) = 19.5*(this->monoDepthMap.at<float>(currentRow, currentCol)) + 0.5;
	    	}

			averageDepthMono = this->monoDepthMap.at<float>(currentRow, currentCol) + averageDepthMono;

		}
	}


	averageDepthMono = averageDepthMono /(rowsInputMap*colsInputMap);
	cv::minMaxLoc(this->confidenceMap, &min, &max);
	double weightNormalized;
	float  cnnZedNormalized;
	float  scaleDivider;

	if(!janivanecky)
		scaleDivider = 20.0;
	else
		scaleDivider = 10.0;

	cv::Mat errorMat(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);
	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{
			int weight = (int) this->confidenceMap.at<uchar>(currentRow, currentCol);
		//	weightNormalized =  computeWeight((double)weight,(double)  max);

			if(this->edgeMap.at<uchar>(currentRow, currentCol) == 255){
				weightNormalized = 1.0; 
			}

			else
				weightNormalized = this->computeWeightBasedOnDistanceToEdge(currentRow,currentCol);

			mergedDepthMap2.at<float>(currentRow, currentCol) =  this->stereoDepthMap.at<float>(currentRow, currentCol)*weightNormalized + (1-weightNormalized)*this->monoDepthMap.at<float>(currentRow, currentCol);

			if(this->monoDepthMap.at<float>(currentRow, currentCol) > this->stereoDepthMap.at<float>(currentRow, currentCol) ){
				cnnZedNormalized = computeWeight( this->stereoDepthMap.at<float>(currentRow, currentCol)/20.0, (float) this->monoDepthMap.at<float>(currentRow, currentCol)/scaleDivider);

			}

			else{
				cnnZedNormalized = computeWeight( (float)this->monoDepthMap.at<float>(currentRow, currentCol)/scaleDivider,  this->stereoDepthMap.at<float>(currentRow, currentCol)/20.0);
			}
				errorMat.at<float>(currentRow, currentCol) = cnnZedNormalized;
				mergedDepthMap.at<float>(currentRow, currentCol) = this->stereoDepthMap.at<float>(currentRow, currentCol)*weightNormalized + (1-weightNormalized)*(  (1- cnnZedNormalized)*this->monoDepthMap.at<float>(currentRow, currentCol)+  this->stereoDepthMap.at<float>(currentRow, currentCol)*cnnZedNormalized );
		}
	}

	cv::minMaxLoc(errorMat, &min, &max);
	errorMat = errorMat/max;
	cv::imshow("weights", errorMat);

	mergedDepthMap2.copyTo(this->secondMap);
	averageDepthMono = averageDepthMono /(rowsInputMap*colsInputMap);
	mergedDepthMap.copyTo(this->finalDepthMap);

	/// Global Variables
	int DELAY_CAPTION = 1500;
	int DELAY_BLUR = 100;
	int MAX_KERNEL_LENGTH = 6;
	cv::Mat copys;
	this->finalDepthMap.copyTo(copys);
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { 
         	bilateralFilter (copys, this->finalDepthMap, i, i*2, i/2 );
         }
}

float confidenceMergeAlgorithm::computeWeightBasedOnDistanceToEdge(int row, int col){

	int counter = 1;
	int searchRegion = 10;

	if(row < searchRegion || col < searchRegion || (row + searchRegion) > this->edgeMap.rows || (col + searchRegion) > this->edgeMap.cols  )
		return(0.0);

	else{

		for (int i = 0; i < searchRegion; ++i)
		{
			if(this->edgeMap.at<uchar>(row-counter, col) == 255 || this->edgeMap.at<uchar>(row, col-counter) == 255 || this->edgeMap.at<uchar>(row+counter, col) == 255 || this->edgeMap.at<uchar>(row, col+counter) == 255)
				return(1/counter);
			
			counter++;
		}
	}

	return(0.0);
}


double confidenceMergeAlgorithm::computeWeight(double conf, double maxConf){
	
	//return(1/(1+exp(-0.5*(10*conf/maxConf-5))));

	return(conf/maxConf);
	/*if(conf/maxConf > 0.5)
		return(1.0);
	else
		return(0.0);*/
};

void confidenceMergeAlgorithm::setmonoDepthMap(cv::Mat inputMonoDepthMap){

	this->monoDepthMap.create(inputMonoDepthMap.rows, inputMonoDepthMap.cols, CV_32FC1);
	inputMonoDepthMap.copyTo(this->monoDepthMap);

};


void confidenceMergeAlgorithm::setstereoDepthMap(cv::Mat stereoInputDepthMap){

	this->stereoDepthMap.create(stereoInputDepthMap.rows, stereoInputDepthMap.cols, CV_32FC1);
	stereoInputDepthMap.copyTo(this->stereoDepthMap);

};

void confidenceMergeAlgorithm::setConfidenceMap(cv::Mat inputconfidenceMap){

	this->confidenceMap.create(inputconfidenceMap.rows, inputconfidenceMap.cols, CV_32FC1);
	inputconfidenceMap.copyTo(this->confidenceMap);

};


void confidenceMergeAlgorithm::setEdgeMap(cv::Mat inputEdgeMap){

	this->edgeMap.create(inputEdgeMap.rows, inputEdgeMap.cols, CV_32FC1);
	inputEdgeMap.copyTo(this->edgeMap);

};

cv::Mat confidenceMergeAlgorithm::getSecondMap(){
	return(this->secondMap);
};

cv::Mat confidenceMergeAlgorithm::getFinalDepthMap(){
	return(this->finalDepthMap);
};

void confidenceMergeAlgorithm::setScaleMonoDepthMap(float newScale){

	this->scaleMonoDepthMap = newScale;

}

void confidenceMergeAlgorithm::setScaleStereoDepthMap(float newScale){

	this->scaleStereoDepthMap = newScale;

}
