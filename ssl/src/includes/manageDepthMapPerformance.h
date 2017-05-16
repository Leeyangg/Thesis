//opencv includes
#include "opencv2/opencv.hpp"

//C includes
#include <math.h>

class manageDepthMapPerformance
{
public:
	manageDepthMapPerformance();
	~manageDepthMapPerformance();
	void setDepthMapGroundTruth(cv::Mat groundTruthMap);
	void setDepthMapEstimation(cv::Mat estimationMap);
	void computePerformance();
	float getThresholdError();
	float getAbsoluteRelativeError();
	float getSquaredRelativeError();
	float getLinearRMSE();
	float getLogRMSE();
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
	float logRMSE;
	float scaleInvariantError;
	float currentPixelGroundTruth;
	float currentPixelPrediction;
	float thresholdErrorThreshold;
	float computeThresholdError();
	float computeAbsoluteRelativeError();
	float computeSquaredRelativeError();
	float computeLinearRMSE();
	float computeLogRMSE();
	void computeScaleInvariantError();

	struct scaleInvariantErrorMetric
	{
		float di       = 0.0;
		float partial1 = 0.0;
		float partial2 = 0.0;
		float lambda   = 0.5;

	};

	scaleInvariantErrorMetric scaleInvariantErrorStruct;

};