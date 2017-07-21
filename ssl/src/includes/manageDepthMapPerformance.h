//opencv includes
#include "opencv2/opencv.hpp"

//C includes
#include <math.h>


//My includes
#include "loadConfiguration.h"

#define ZED_BIAS 0.5

class manageDepthMapPerformance
{
public:
	manageDepthMapPerformance();
	~manageDepthMapPerformance();
	void setDepthMapGroundTruth(cv::Mat groundTruthMap);
	void setDepthMapEstimation(cv::Mat estimationMap);
	void computePerformance();
	void computePerformance(cv::Mat confidenceMap);
	float getThresholdError();
	float getAbsoluteRelativeError();
	float getSquaredRelativeError();
	float getLinearRMSE();
	double getLogRMSE();
	float getScaleInvariantError();
	void setThresholdErrorThreshold(float threshold);
	void setScaleDepthMap(float scale);
	void setScaleGroundTruth(float scale);

private:
	cv::Mat groundTruthMap;
	cv::Mat estimationMap;
	float scaleDepthMap;
	float scaleGroundTruth;
	float thresholdError ;
	float absoluteRelativeError;
	float squaredRelativeError;
	float linearRMSE;
	double logRMSE;
	float scaleInvariantError;
	float currentPixelGroundTruth;
	float currentPixelPrediction;
	float thresholdErrorThreshold;
	float computeThresholdError();
	float computeAbsoluteRelativeError();
	float computeSquaredRelativeError();
	float computeLinearRMSE();
	double computeLogRMSE();
	float computeScaleInvariantError();
	void computePartialsScaleInvariantError();
	void resetErrors();


	struct scaleInvariantErrorMetric
	{
		float di       = 0.0;
		float partial1 = 0.0;
		std::vector<float> partial2;
		float lambda   = 0.5;

	};

	scaleInvariantErrorMetric scaleInvariantErrorStruct;

};
