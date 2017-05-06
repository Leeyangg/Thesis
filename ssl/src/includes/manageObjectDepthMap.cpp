#include "manageObjectDepthMap.h"

manageObjectDepthMap::manageObjectDepthMap(){}
manageObjectDepthMap::~manageObjectDepthMap(){}

void manageObjectDepthMap::setDepthMap(cv::Mat referenceMap){

	referenceMap.copyTo(this->depthMap);

}

void manageObjectDepthMap::mergeDepthMap(cv::Mat map2MergeWith, std::string method){

	if(strcmp("facil", method.c_str()) == 0){
			
		this->merger.setmonoDepthMap(this->depthMap);
		this->merger.setstereoDepthMap(map2MergeWith);
		this->merger.setPixels2BeMerged(this->pixels2BeMerged);

		if(this->pixels2BeMerged.size() > 0)
			this->merger.facilOriginal();
		
		this->mergedDepthMap = merger.getFinalDepthMap();
		this->merger.pixels2BeMerged.clear();

	}

}

cv::Mat manageObjectDepthMap::getMergedDepthMap(){

	return(this->mergedDepthMap);

}

void manageObjectDepthMap::filterPixels2BeMerged(cv::Mat referenceMap){

	referenceMap.convertTo(referenceMap, CV_8UC1);
	int rowsInputMap = referenceMap.rows;
	int colsInputMap = referenceMap.cols;
	cv::Point_<int> addPixel2Fusion;

	for(int currentRow = 0; currentRow < rowsInputMap; currentRow++){
		for(int currentCol = 0; currentCol < colsInputMap; currentCol++){

			int confidenceLevel = referenceMap.data[currentRow*referenceMap.step + currentCol*referenceMap.elemSize()]; 

			if(confidenceLevel < this->thresholdFilter) {

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
	this->colorMap.create(this->mapResolution, CV_32FC3);
	this->scaleFactor = 1/10;

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
			applyColorMap(this->map, this->colorMap,  cv::COLORMAP_BONE);
			break;

		default:
			applyColorMap(this->map, this->colorMap,  cv::COLORMAP_RAINBOW);
			break;
	}
}

void displayObjectDepthMap::saveMap(){

	cv::imwrite("/" + this->windowTitle + std::to_string(this->saveCounter) + ".jpeg", map);

}

void displayObjectDepthMap::setScaleFactor(float newScale){

	this->scaleFactor = newScale;

}
