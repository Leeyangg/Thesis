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

class manageZEDObject
{
public:
	manageZEDObject();
	~manageZEDObject();
	void setMaximumDepthDistance(float maximumDepth);
	cv::Mat getImage();
	cv::Mat getConfidenceMap();
	cv::Mat getDepthMap();
	cv::Mat getLeftImage(bool save);
	cv::Mat getRightImage(bool save);
	void grabFrame();
	
protected:

private:
#ifdef COMPILE_ZED
	sl::zed::Camera* zedObject;
	sl::zed::SENSING_MODE dm_type = sl::zed::FULL;
#endif

	int currentFrame = 0;
	float maximumZedDepth;
	void checkZEDStart();
    int mapWidth;
	int mapHeight;
	float scaleToConvertMapToMeters;
	float maximumDepth;
	cv::Mat zedCVImage;
    cv::Mat zedCVMap;
	void setMapWidth();
	void setMapHeight();
	cv::VideoCapture zedOpencv;
	cv::Mat rightImage;
	cv::Mat leftImage;

};

void grabFrameZed(manageZEDObject* zedCamObject);
