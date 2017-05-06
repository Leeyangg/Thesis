// OpenCV includes
#include "opencv2/opencv.hpp"

class facilMergeAlgorithm
 {

 public:

 	facilMergeAlgorithm();
 	~facilMergeAlgorithm();
 	void setmonoDepthMap(cv::Mat inputMonoDepthMap);
 	void setstereoDepthMap(cv::Mat inputStereoDepthMap);
 	void setPixels2BeMerged(std::vector<cv::Point_<int>> inputPixels2BeMerged);
 	cv::Mat getFinalDepthMap();
 	void facilOriginal();
 	void setSigma1(int newSigma);
 	void setSigma2(int newSigma);
 	void setSigma3(int newSigma);
 	std::vector<cv::Point_<int>> pixels2BeMerged;
 	void setScaleMonoDepthMap(int newScale);
 	void setScaleStereoDepthMap(int newScale);

 private:

 	cv::Mat monoDepthMap;
 	cv::Mat stereoDepthMap;
 	cv::Mat finalDepthMap;
 	cv::Mat derivativeX;
 	cv::Mat derivativeY;
 	float sigma1 = 15.0;
 	float sigma2 = 0.1;
 	float sigma3 = 1*exp(-3);
 	float scaleMonoDepthMap = 10;
 	float scaleStereoDepthMap = 10/255;
 	void computeXDerivative();
 	void computeYDerivative();
 	float computeW1(cv::Point_<int> currentPixel, int currentPointFromSparse);
 	float computeW2(cv::Point_<int> currentPixel, int currentPointFromSparse);
 	float computeW3(cv::Point_<int> currentPixel, int currentPointFromSparse);
 	float computeW4(cv::Point_<int> currentPixel, int currentPointFromSparse);
  	float normalizeWeights(float partialWeight, float minAllPartialWeights, float sumOfAllPartialWeights );
	float computeDepth(cv::Point_<int> currentPixel);
 	
 };