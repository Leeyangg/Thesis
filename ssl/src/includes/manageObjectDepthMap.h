//Opencv includes
#include "opencv2/opencv.hpp"

//My includes
#include "facilMergeAlgorithm.h"

class manageObjectDepthMap
{
public:
	manageObjectDepthMap();
	~manageObjectDepthMap();
	cv::Mat getMergedDepthMap();
	cv::Mat getSecondMap();
	void mergeDepthMap(cv::Mat map2MergeWith, std::string method,  float scaleInputDepthMap, float scaleSSLCnnMap);
	void filterPixels2BeMerged(cv::Mat referenceMap);
	void filterPixels2BeMerged();
	void setThresholdFilter(int threshold);
	int  getThresholdFilter();
	void refreshPixels2BeMerged();
	void setDepthMap(cv::Mat referenceMap);

private:
	cv::Mat mergedDepthMap;
	cv::Mat secondMap;
	cv::Mat depthMap;
	std::vector<cv::Point_<int>> pixels2BeMerged;
	int thresholdFilter;
	facilMergeAlgorithm  merger;
	float scale;

};

class displayObjectDepthMap
{
public:
	displayObjectDepthMap();
	~displayObjectDepthMap();
	void setMapResolution(cv::Size newResolution);
	void setScaleFactor(float newScale);
	void displayMat();
	void displayColorMat();
	void useColorMap(int choiceMap);
	void saveMap();
	void setMap(cv::Mat map, std::string windowTitle);

private:
	cv::Size mapResolution;
	cv::Mat map;
	cv::Mat colorMap;
	std::string windowTitle;
	int saveCounter;
	float scaleFactor;
};
