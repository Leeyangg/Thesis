//My includes
#include "facilMergeAlgorithm.h"

facilMergeAlgorithm::facilMergeAlgorithm(){}

facilMergeAlgorithm::~facilMergeAlgorithm(){}

void facilMergeAlgorithm::facilOriginal(){

   this->stereoDepthMap.convertTo(this->stereoDepthMap, CV_32FC1);
   this->monoDepthMap.convertTo(this->monoDepthMap, CV_32FC1);
	this->finalDepthMap.create(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);
	this->secondMap.create(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);

	float averageDepthMono = 0.0;
	cv::Mat mergedDepthMap(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1, 0.5);
	int rowsInputMap = this->monoDepthMap.rows;
	int colsInputMap = this->monoDepthMap.cols;
	cv::Point_<int> currentPixel;

	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{

			this->stereoDepthMap.at<float>(currentRow, currentCol) = this->stereoDepthMap.at<float>(currentRow, currentCol)*this->scaleStereoDepthMap;
			this->monoDepthMap.at<float>(currentRow, currentCol) = this->monoDepthMap.at<float>(currentRow, currentCol)*this->scaleMonoDepthMap;
			averageDepthMono = this->monoDepthMap.at<float>(currentRow, currentCol) + averageDepthMono;

		}
	
	}

	averageDepthMono = averageDepthMono /(rowsInputMap*colsInputMap);

	this->computeXDerivative();
	this->computeYDerivative();

	for (int currentRow = 0; currentRow < rowsInputMap; ++currentRow)
	{
		for (int currentCol = 0; currentCol < colsInputMap; ++currentCol)
		{
			currentPixel.x = currentCol;
			currentPixel.y = currentRow;
			this->col = currentCol;
			this->row = currentRow;
			mergedDepthMap.at<float>(currentRow, currentCol) = this->computeDepth(currentPixel);

			if(mergedDepthMap.at<float>(currentRow, currentCol) < 0)
				mergedDepthMap.at<float>(currentRow, currentCol) = averageDepthMono;

		}
	
	}
	mergedDepthMap.copyTo(this->finalDepthMap);
}

float facilMergeAlgorithm::computeDepth(cv::Point_<int> currentPixelMono){

	int numberOfPixels2Merge = this->pixels2BeMerged.size();
	std::vector<float> partialWeights(4);
	float allPartialWeights[this->pixels2BeMerged.size()] ;
	float sumOfAllPartialWeights = 0.0;
	float minAllPartialWeights = 0.0;
	bool firstPartialWeight = true;
	float pixelDepth = 0.0;
	float currentPartialWeight = 0.0;

	for (int currentPointFromSparse = 0; currentPointFromSparse < numberOfPixels2Merge; ++currentPointFromSparse)
	{
		  partialWeights[0] = this->computeW1(currentPixelMono, currentPointFromSparse);
		  partialWeights[1] = this->computeW2(currentPixelMono, currentPointFromSparse);
        partialWeights[2] = this->computeW3(currentPixelMono, currentPointFromSparse);
        partialWeights[3] = this->computeW4(currentPixelMono, currentPointFromSparse);

        currentPartialWeight = (partialWeights[0]*partialWeights[1]*partialWeights[2]*partialWeights[3]);
        allPartialWeights[currentPointFromSparse] = (partialWeights[0]*partialWeights[1]*partialWeights[2]*partialWeights[3]);
		
        sumOfAllPartialWeights = sumOfAllPartialWeights + currentPartialWeight;

        if(firstPartialWeight){
        	minAllPartialWeights = currentPartialWeight;
        	firstPartialWeight = false;
        }

        else{
        	if(currentPartialWeight < minAllPartialWeights)
        		minAllPartialWeights = currentPartialWeight;
        }           
	}

	std::vector<float> vectorNormalizedWeights;
	float sumWeights = 0.0;
	float maxWeight = 0.0;

	for (int currentPointFromSparse2 = 0; currentPointFromSparse2 < numberOfPixels2Merge; ++currentPointFromSparse2)
	{
		float normalizeWeight = 0.0;

		normalizeWeight =  this->normalizeWeights(allPartialWeights[currentPointFromSparse2] , minAllPartialWeights , sumOfAllPartialWeights);

		sumWeights = sumWeights + normalizeWeight;
		vectorNormalizedWeights.push_back(normalizeWeight);

		if(normalizeWeight > maxWeight)
			maxWeight = normalizeWeight;

		pixelDepth = pixelDepth +  normalizeWeight*(this->stereoDepthMap.at<float>(this->pixels2BeMerged[currentPointFromSparse2].y,this->pixels2BeMerged[currentPointFromSparse2].x) + this->monoDepthMap.at<float>(currentPixelMono.y,currentPixelMono.x) - this->monoDepthMap.at<float>(this->pixels2BeMerged[currentPointFromSparse2].y, this->pixels2BeMerged[currentPointFromSparse2].x));
	
	}

	this->meanWeightVector = sumWeights/numberOfPixels2Merge;
	this->computeStdDev(vectorNormalizedWeights);
	float xx =0.0;
	float threshold = this->meanWeightVector + this->stdDevWeightVector;

	for (int currentPointFromSparse2 = 0; currentPointFromSparse2 < numberOfPixels2Merge; ++currentPointFromSparse2)
	{
	
		if(vectorNormalizedWeights[currentPointFromSparse2] > threshold){
		
			xx = xx +  vectorNormalizedWeights[currentPointFromSparse2]*(this->stereoDepthMap.at<float>(this->pixels2BeMerged[currentPointFromSparse2].y,this->pixels2BeMerged[currentPointFromSparse2].x) + this->monoDepthMap.at<float>(currentPixelMono.y,currentPixelMono.x) - this->monoDepthMap.at<float>(this->pixels2BeMerged[currentPointFromSparse2].y, this->pixels2BeMerged[currentPointFromSparse2].x));
			this->secondMap.at<float>(this->row, this->col) = xx;

		}
	
	}



	return (pixelDepth);
}

void facilMergeAlgorithm::computeStdDev(std::vector<float> vectorNormalizedWeights){

	this->stdDevWeightVector = 0.0;

	for(auto normalizedWeight: vectorNormalizedWeights){

		this->stdDevWeightVector = this->stdDevWeightVector + (1.0/(this->pixels2BeMerged.size()-1))*pow(normalizedWeight-this->meanWeightVector,2);

	}

	this->stdDevWeightVector = sqrt(this->stdDevWeightVector);

}

float facilMergeAlgorithm::normalizeWeights(float partialWeight, float minAllPartialWeights, float sumOfAllPartialWeights ){

	int numberOfPixels2Merge = this->pixels2BeMerged.size();
	float normalizeWeights;

	normalizeWeights = (partialWeight - minAllPartialWeights)/(sumOfAllPartialWeights - minAllPartialWeights);

	return(normalizeWeights);

}


float facilMergeAlgorithm::computeW1(cv::Point_<int> currentPixelMono, int currentPointFromSparse){

	return(exp( (-1*sqrt( pow((currentPixelMono.y -  this->pixels2BeMerged[currentPointFromSparse].y),2) +  pow((currentPixelMono.x -  this->pixels2BeMerged[currentPointFromSparse].x),2) ))/this->sigma1 ));

}

float facilMergeAlgorithm::computeW2(cv::Point_<int> currentPixelMono, int currentPointFromSparse){

	return((1/(abs(this->derivativeX.at<float>( this->pixels2BeMerged[currentPointFromSparse].y,  this->pixels2BeMerged[currentPointFromSparse].x) - this->derivativeX.at<float>(currentPixelMono.y,currentPixelMono.x)) + this->sigma2)) * (1/(abs(  this->derivativeY.at<float>( this->pixels2BeMerged[currentPointFromSparse].y,  this->pixels2BeMerged[currentPointFromSparse].x) - this->derivativeY.at<float>(currentPixelMono.y,currentPixelMono.x) ) + this->sigma2)) );

}

float facilMergeAlgorithm::computeW3(cv::Point_<int> currentPixelMono, int currentPointFromSparse){

	return( exp( -abs( this->monoDepthMap.at<float>(currentPixelMono.y,currentPixelMono.x)  + this->derivativeX.at<float>(currentPixelMono.y,currentPixelMono.x)*(currentPixelMono.y -  this->pixels2BeMerged[currentPointFromSparse].y) - this->monoDepthMap.at<float>( this->pixels2BeMerged[currentPointFromSparse].y,  this->pixels2BeMerged[currentPointFromSparse].x)  ) ) + this->sigma3);

}

float facilMergeAlgorithm::computeW4(cv::Point_<int> currentPixelMono, int currentPointFromSparse){

	return(exp( -abs( this->monoDepthMap.at<float>(currentPixelMono.y,currentPixelMono.x) + this->derivativeY.at<float>(currentPixelMono.y,currentPixelMono.x)*(currentPixelMono.y -  this->pixels2BeMerged[currentPointFromSparse].y) - this->monoDepthMap.at<float>( this->pixels2BeMerged[currentPointFromSparse].y,  this->pixels2BeMerged[currentPointFromSparse].x)  ) ) + this->sigma3);


}

void facilMergeAlgorithm::setmonoDepthMap(cv::Mat inputMonoDepthMap){

	this->monoDepthMap.create(inputMonoDepthMap.rows, inputMonoDepthMap.cols, CV_32FC1);
	inputMonoDepthMap.copyTo(this->monoDepthMap);

};


void facilMergeAlgorithm::setstereoDepthMap(cv::Mat stereoInputDepthMap){

	this->stereoDepthMap.create(stereoInputDepthMap.rows, stereoInputDepthMap.cols, CV_32FC1);
	stereoInputDepthMap.copyTo(this->stereoDepthMap);

};

void facilMergeAlgorithm::setPixels2BeMerged(std::vector<cv::Point_<int>> inputPixels2BeMerged){

	this->pixels2BeMerged = inputPixels2BeMerged; 

};

cv::Mat facilMergeAlgorithm::getFinalDepthMap(){

	return(this->finalDepthMap);

};

cv::Mat facilMergeAlgorithm::getSecondMap(){

	return(this->secondMap);

};


void facilMergeAlgorithm::computeXDerivative(){

	this->derivativeX.create(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);
    cv::Sobel(this->monoDepthMap,this->derivativeX,-1, 1, 0, -1, 1, 0, cv::BORDER_DEFAULT); 

}

void facilMergeAlgorithm::computeYDerivative(){

	this->derivativeY.create(this->monoDepthMap.rows,this->monoDepthMap.cols, CV_32FC1);
	cv::Sobel(this->monoDepthMap,this->derivativeY, -1, 0, 1, -1, 1, 0, cv::BORDER_DEFAULT); 
	
}

void facilMergeAlgorithm::setSigma1(int newSigma){

	this->sigma1= newSigma;

}

void facilMergeAlgorithm::setSigma2(int newSigma){

	this->sigma2= newSigma;
	
}

void facilMergeAlgorithm::setSigma3(int newSigma){

	this->sigma3= newSigma;
	
}

void facilMergeAlgorithm::setScaleMonoDepthMap(float newScale){

	this->scaleMonoDepthMap = newScale;

}

void facilMergeAlgorithm::setScaleStereoDepthMap(float newScale){

	this->scaleStereoDepthMap = newScale;

}
