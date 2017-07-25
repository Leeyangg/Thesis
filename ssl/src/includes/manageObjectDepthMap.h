//Opencv includes
#include "opencv2/opencv.hpp"

//My includes
#include "facilMergeAlgorithm.h"
#include "confidenceMergeAlgorithm.h"

class manageObjectDepthMap
{
public:
	manageObjectDepthMap();
	~manageObjectDepthMap();
	cv::Mat getMergedDepthMap();
	cv::Mat getSecondMap();
	void mergeDepthMap(cv::Mat map2MergeWith, std::string method,  float scaleInputDepthMap, float scaleSSLCnnMap);
	void filterPixels2BeMerged(cv::Mat referenceMap);
	void filterPixels2BeMerged(cv::Mat referenceMap, int threshold);
	void filterPixels2BeMerged();
	void setThresholdFilter(int threshold);
	int  getThresholdFilter();
	int getNumberPixelsForMerge();
	void refreshPixels2BeMerged();
	void setDepthMap(cv::Mat referenceMap);
	void setConfidenceMap(cv::Mat referenceMap);
	void setEdgeMap(cv::Mat referenceMap);
	cv::Mat getPointsForSSL();

private:
	cv::Mat mergedDepthMap;
	cv::Mat secondMap;
	cv::Mat depthMap;
	cv::Mat confidenceMap;
	cv::Mat edgeMap;
	std::vector<cv::Point_<int>> pixels2BeMerged;
	int thresholdFilter;
	facilMergeAlgorithm  merger;
	confidenceMergeAlgorithm confMerger;
	cv::Mat pointsForSSL;
	float scale;
	int availablePixelFromSparse;

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
	void saveMap(int frame,std::string name);
	void setMap(cv::Mat map, std::string windowTitle);

private:
	cv::Size mapResolution;
	cv::Mat map;
	cv::Mat colorMap;
	std::string windowTitle;
	int saveCounter;
	float scaleFactor;
};
