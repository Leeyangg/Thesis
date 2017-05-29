#ifdef COMPILE_ZED
	//ZED Includes
	#include <zed/Camera.hpp>
	#include <zed/utils/GlobalDefine.hpp>
#endif

// Opencv includes
#include "opencv2/opencv.hpp"

// C includes
#include <stdlib.h>
#include <thread>
#include <vector>

class manageZEDObject
{
public:
	manageZEDObject();
	~manageZEDObject();
	void setMaximumDepthDistance(float maximumDepth);
	void grabFrame();
	std::vector<cv::Mat> getImage();
	cv::Mat getConfidenceMap();
	cv::Mat getDepthMap();
	cv::Mat getLeftImage(bool save);
	cv::Mat getRightImage(bool save);
	
protected:

private:
#ifdef COMPILE_ZED
	sl::zed::Camera* zedObject;
	sl::zed::SENSING_MODE dm_type = sl::zed::FULL;
#endif

	int currentFrame = 0;
   int mapWidth;
	int mapHeight;
	float maximumZedDepth;
	float scaleToConvertMapToMeters;
	float maximumDepth;	
	cv::Mat zedCVImage;
	cv::Mat zedCVImage2;
   cv::Mat zedCVMap;
	cv::VideoCapture zedOpencv;
	cv::Mat rightImage;
	cv::Mat leftImage;
	void checkZEDStart();
	void setMapWidth();
	void setMapHeight();

};

void grabFrameZed(manageZEDObject* zedCamObject);
