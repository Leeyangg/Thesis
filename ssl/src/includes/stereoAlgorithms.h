//Opencv includes
#include "opencv2/opencv.hpp"

class stereoBMOpencv
{
public:
	stereoBMOpencv();
	~stereoBMOpencv();
	void setLeftImage(cv::Mat image);
	void setRightImage(cv::Mat image);
	void computeDisparityMap();
	void computeAbsoluteDepthMap();
	void setScaleDepthMap(float scale);
	void setResolution(cv::Size resolution);
	int getAvailablePoints();
	cv::Mat getAbsoluteDepthMap();
	cv::Mat getAbsoluteDepthMapResized();
	cv::Mat getDisparityMap();
	cv::Mat getDisparityMapResized();
	cv::Mat getPointsForSSL();

private:
	float scaleDepthMap;
	int availablePoints;
	cv::Size resolution;
	cv::Mat leftImage;
	cv::Mat leftImageGrayScale;
	cv::Mat rightImage;
	cv::Mat rightImageGrayScale;
	cv::Mat disparityMap;
	cv::Mat disparityMapResized;
	cv::Mat absoluteDepthMap;
	cv::Mat absoluteDepthMapResized;
	cv::Mat pointsForSSL;
	cv::StereoBM stereoBM;
	cv::StereoSGBM stereoSGBM;
};


