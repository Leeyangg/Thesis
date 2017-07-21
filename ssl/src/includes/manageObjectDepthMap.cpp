#include "manageObjectDepthMap.h"

extern  cv::Size resolutionOutputMapsJSONFile;

manageObjectDepthMap::manageObjectDepthMap(){

	this->thresholdFilter = 30;	
	this->availablePixelFromSparse =0;

}
manageObjectDepthMap::~manageObjectDepthMap(){}

void manageObjectDepthMap::setDepthMap(cv::Mat referenceMap){
	referenceMap.copyTo(this->depthMap);
}

void manageObjectDepthMap::setConfidenceMap(cv::Mat referenceMap){
	referenceMap.copyTo(this->confidenceMap);
}

void manageObjectDepthMap::setEdgeMap(cv::Mat referenceMap){
	referenceMap.copyTo(this->edgeMap);
}


void manageObjectDepthMap::mergeDepthMap(cv::Mat map2MergeWith, std::string method, float scaleInputDepthMap, float scaleSSLCnnMap){

	if(strcmp("facil", method.c_str()) == 0){
		
		this->merger.setmonoDepthMap(this->depthMap);
		this->merger.setstereoDepthMap(map2MergeWith);
		this->merger.setPixels2BeMerged(this->pixels2BeMerged);
		this->merger.setScaleMonoDepthMap(scaleSSLCnnMap);
		this->merger.setScaleStereoDepthMap(scaleInputDepthMap);

		if(this->pixels2BeMerged.size() > 0){
			this->merger.facilOriginal();
			this->mergedDepthMap = merger.getFinalDepthMap();
			this->secondMap = merger.getSecondMap();
		}

		else{
			this->depthMap.copyTo(this->mergedDepthMap);
			this->depthMap.copyTo(this->secondMap);
		}

		this->merger.pixels2BeMerged.clear();

	}


	else if(strcmp("confMerger", method.c_str()) == 0){

		this->confMerger.setmonoDepthMap(this->depthMap);
		this->confMerger.setstereoDepthMap(map2MergeWith);		
		this->confMerger.setConfidenceMap(this->confidenceMap);
		this->confMerger.setEdgeMap(this->edgeMap);			
		this->confMerger.setScaleMonoDepthMap(scaleSSLCnnMap);		
		this->confMerger.setScaleStereoDepthMap(scaleInputDepthMap);

		this->confMerger.merge();
		this->mergedDepthMap =this->confMerger.getFinalDepthMap();
		this->secondMap =this->confMerger.getSecondMap();
		
	}

}

cv::Mat manageObjectDepthMap::getMergedDepthMap(){

	return(this->mergedDepthMap);

}

cv::Mat manageObjectDepthMap::getSecondMap(){

	return(this->secondMap);

}


cv::Mat manageObjectDepthMap::getPointsForSSL(){
	return(this->pointsForSSL);
}

int manageObjectDepthMap::getNumberPixelsForMerge(){
	return(this->availablePixelFromSparse);
}

void manageObjectDepthMap::filterPixels2BeMerged(cv::Mat referenceMap, int threshold){

	this->availablePixelFromSparse = 0;
	referenceMap.convertTo(referenceMap, CV_32FC1);
	this->pointsForSSL.create(referenceMap.rows, referenceMap.cols, CV_32FC1);
	int rowsInputMap = referenceMap.rows;
	int colsInputMap = referenceMap.cols;
	cv::Point_<int> addPixel2Fusion;

	for(int currentRow = 0; currentRow < rowsInputMap; currentRow++){
		for(int currentCol = 0; currentCol < colsInputMap; currentCol++){
			if(referenceMap.at<float>(currentRow, currentCol) >= threshold ) {
				addPixel2Fusion.x = currentCol;
				addPixel2Fusion.y = currentRow;
				(this->pixels2BeMerged).push_back(addPixel2Fusion);
				this->pointsForSSL.at<float>(currentRow, currentCol) = 1.0;
				this->availablePixelFromSparse++;
			}

			else
				this->pointsForSSL.at<float>(currentRow, currentCol) = 0.0;
		}
	}
}

void manageObjectDepthMap::filterPixels2BeMerged(cv::Mat referenceMap){

	this->availablePixelFromSparse = 0;
	cv::Mat referenceMapResized;
	cv::resize(referenceMap, referenceMapResized, resolutionOutputMapsJSONFile);
	this->pointsForSSL.create(referenceMapResized.rows, referenceMapResized.cols, CV_32FC1);
	int rowsInputMap = referenceMapResized.rows;
	int colsInputMap = referenceMapResized.cols;
	cv::Point_<int> addPixel2Fusion;

	for(int currentRow = 0; currentRow < rowsInputMap; currentRow++){
		for(int currentCol = 0; currentCol < colsInputMap; currentCol++){

			if(referenceMapResized.at<float>(currentRow, currentCol) > 0.0 ) {
				addPixel2Fusion.x = currentCol;
				addPixel2Fusion.y = currentRow;
				(this->pixels2BeMerged).push_back(addPixel2Fusion);
				this->pointsForSSL.at<float>(currentRow, currentCol) = 1.0;
				this->availablePixelFromSparse++;
			}

			else
				this->pointsForSSL.at<float>(currentRow, currentCol) = 0.0;
		}
	}
}

void manageObjectDepthMap::filterPixels2BeMerged(){

	int rowsInputMap = this->depthMap.rows;
	int colsInputMap = this->depthMap.cols;
	cv::Point_<int> addPixel2Fusion;

	for(int currentRow = 0; currentRow < rowsInputMap; currentRow++){
		for(int currentCol = 0; currentCol < colsInputMap; currentCol++){

			if( remainder(currentRow,8)==0 && remainder(currentCol,8)==0){

				addPixel2Fusion.x = currentCol;
				addPixel2Fusion.y = currentRow;
				(this->pixels2BeMerged).push_back(addPixel2Fusion);

			}
		}
	}
}


void manageObjectDepthMap::setThresholdFilter(int threshold){

	this->thresholdFilter = threshold;

}

int manageObjectDepthMap::getThresholdFilter(){

	return(this->thresholdFilter);

}

void manageObjectDepthMap::refreshPixels2BeMerged(){

	this->pixels2BeMerged.clear();

}



displayObjectDepthMap::displayObjectDepthMap(){

	this->saveCounter = 0;
	this->mapResolution.height = 160;
	this->mapResolution.width = 256;
	this->map.create(this->mapResolution, CV_32FC3);
	this->colorMap.create(this->mapResolution, CV_32FC3);
	this->scaleFactor = 1;

}

displayObjectDepthMap::~displayObjectDepthMap(){}

void displayObjectDepthMap::setMap(cv::Mat map, std::string windowTitle){

	cv::resize(map, this->map,this->mapResolution);

	this->windowTitle.assign(windowTitle);

}

void displayObjectDepthMap::setMapResolution(cv::Size newResolution){

	this->mapResolution.height = newResolution.height;
	this->mapResolution.width = newResolution.width;

}


void displayObjectDepthMap::displayMat(){

	cv::imshow(this->windowTitle, this->map);

}

void displayObjectDepthMap::displayColorMat(){

	cv::imshow(this->windowTitle, this->colorMap);

}

void displayObjectDepthMap::useColorMap(int choiceMap){

	cv::convertScaleAbs(this->map, this->map, this->scaleFactor);
	
	switch (choiceMap){
		case 1:
			applyColorMap(this->map, this->colorMap,  cv::COLORMAP_RAINBOW);
			break;

		case 2: 
			applyColorMap(this->map, this->colorMap,  cv::COLORMAP_JET);
			break;

		default:
			this->map.copyTo(this->colorMap);
			break;
	}
}

void displayObjectDepthMap::saveMap(int frame){

	cv::imwrite("./images/" + this->windowTitle + std::to_string(frame) + ".jpg", this->colorMap);

}

void displayObjectDepthMap::setScaleFactor(float newScale){

	this->scaleFactor = newScale;

}
