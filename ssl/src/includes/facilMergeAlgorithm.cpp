//My includes
#include "facilMergeAlgorithm.h"

extern bool stereoOpenCVJSONFile;

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

			if(stereoOpenCVJSONFile)
				this->stereoDepthMap.at<float>(currentRow, currentCol) = this->stereoDepthMap.at<float>(currentRow, currentCol)*this->scaleStereoDepthMap;

			else
				this->stereoDepthMap.at<float>(currentRow, currentCol) = this->stereoDepthMap.at<float>(currentRow, currentCol)*(19.5/255.0*(-1.0))+20.0;

			//this->monoDepthMap.at<float>(currentRow, currentCol) = this->monoDepthMap.at<float>(currentRow, currentCol)*this->scaleMonoDepthMap + BIAS_GT;

	    	if(!janivanecky)
	    		this->monoDepthMap.at<float>(currentRow, currentCol) = this->monoDepthMap.at<float>(currentRow, currentCol)*(19.5*(-1.0))+20.0;

	    	else{
	    		//CHECK IF I NEED TO DIVIDE BY 255555555555555!!!!!!!!!!!!!!!
	    		this->monoDepthMap.at<float>(currentRow, currentCol) = 19.5*this->monoDepthMap.at<float>(currentRow, currentCol)/255.0 + 0.5;

	    		if(this->monoDepthMap.at<float>(currentRow, currentCol) == 0)
	    			this->monoDepthMap.at<float>(currentRow, currentCol) = 0.5;
	    	}

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

			if(mergedDepthMap.at<float>(currentRow, currentCol) <= 0){
				mergedDepthMap.at<float>(currentRow, currentCol) = BIAS_GT;
			}

			if(this->secondMap.at<float>(this->row, this->col) <= 0){
				this->secondMap.at<float>(this->row, this->col) = BIAS_GT;
			}

		}
	
	}
	mergedDepthMap.copyTo(this->finalDepthMap);


	/// Global Variables
	int DELAY_CAPTION = 1500;
	int DELAY_BLUR = 100;
	int MAX_KERNEL_LENGTH = 6;
/*
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        { cv::GaussianBlur( this->secondMap, this->secondMap, cv::Size( i, i ), 0, 0 );
         }
*/
	cv::Mat copys;
	 this->secondMap.copyTo(copys);
     for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { bilateralFilter (copys, this->secondMap, i, i*2, i/2 );}
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
	float depthSecondMap = 0.0;
	float threshold = this->meanWeightVector + 2*this->stdDevWeightVector;
	sumOfAllPartialWeights = 0.0;
	minAllPartialWeights = 0.0;
	std::vector<cv::Point_<int>> newPixels2BeMerged;
	firstPartialWeight = true;
	std::vector<float> newAllPartialWeights;

	for (int currentPointFromSparse2 = 0; currentPointFromSparse2 < numberOfPixels2Merge; ++currentPointFromSparse2)
	{
	
		if(vectorNormalizedWeights[currentPointFromSparse2] > threshold){
							depthSecondMap = depthSecondMap +  vectorNormalizedWeights[currentPointFromSparse2]*(this->stereoDepthMap.at<float>(this->pixels2BeMerged[currentPointFromSparse2].y,this->pixels2BeMerged[currentPointFromSparse2].x) + this->monoDepthMap.at<float>(currentPixelMono.y,currentPixelMono.x) - this->monoDepthMap.at<float>(this->pixels2BeMerged[currentPointFromSparse2].y,this->pixels2BeMerged[currentPointFromSparse2].x));
		}}/*

			newPixels2BeMerged.push_back(this->pixels2BeMerged.at(currentPointFromSparse2));
	        sumOfAllPartialWeights = sumOfAllPartialWeights + allPartialWeights[currentPointFromSparse2];
			newAllPartialWeights.push_back(allPartialWeights[currentPointFromSparse2]);
        
	        if(firstPartialWeight){
	        	minAllPartialWeights = allPartialWeights[currentPointFromSparse2];
	        	firstPartialWeight = false;
	        }

	        else{
	        	if(allPartialWeights[currentPointFromSparse2] < minAllPartialWeights)
	        		minAllPartialWeights = allPartialWeights[currentPointFromSparse2];
	        }  	

		}
	}

	if(newPixels2BeMerged.size() > 1){

		for (int currentPointFromSparse2 = 0; currentPointFromSparse2 < newPixels2BeMerged.size(); ++currentPointFromSparse2)
		{
				float normalizeWeight = 0.0;
				normalizeWeight =  this->normalizeWeights( newAllPartialWeights[currentPointFromSparse2] , minAllPartialWeights , sumOfAllPartialWeights);
				depthSecondMap = depthSecondMap +  normalizeWeight*(this->stereoDepthMap.at<float>(newPixels2BeMerged[currentPointFromSparse2].y,newPixels2BeMerged[currentPointFromSparse2].x) + this->monoDepthMap.at<float>(currentPixelMono.y,currentPixelMono.x) - this->monoDepthMap.at<float>(newPixels2BeMerged[currentPointFromSparse2].y,newPixels2BeMerged[currentPointFromSparse2].x));
				
		}
	}
	else{
		depthSecondMap = this->stereoDepthMap.at<float>(this->row, this->col);
	}
*/
	this->secondMap.at<float>(this->row, this->col) = depthSecondMap;
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

	if(sumOfAllPartialWeights != minAllPartialWeights)
		normalizeWeights = (partialWeight - minAllPartialWeights)/(sumOfAllPartialWeights - minAllPartialWeights);
	else
		normalizeWeights = 1.0;
		
			//	std::cout << normalizeWeights << " " << partialWeight << " " << minAllPartialWeights << " " <<  sumOfAllPartialWeights << " "  << "\n";

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
