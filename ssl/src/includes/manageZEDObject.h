#ifdef COMPILE_ZED
	//ZED Includes
	#include <zed/Camera.hpp>
	#include <zed/utils/GlobalDefine.hpp>
#endif

// Opencv includes
#include "opencv2/opencv.hpp"

// C includes
#include <stdlib.h>


class manageZEDObject
{
public:
	manageZEDObject();
	~manageZEDObject();
	void setMaximumDepthDistance(float maximumDepth);
	cv::Mat getImage();
	cv::Mat getConfidenceMap();
	cv::Mat getDepthMap();
	void grabFrame();
	
protected:

private:
#ifdef COMPILE_ZED
	sl::zed::Camera* zedObject;
	sl::zed::SENSING_MODE dm_type = sl::zed::FULL;
#endif


	float maximumZedDepth;
	void checkZEDStart();
    float mapWidth;
	float mapHeight;
	float scaleToConvertMapToMeters;
	float maximumDepth;
	cv::Mat zedCVImage;
    cv::Mat zedCVMap;
	void setMapWidth();
	void setMapHeight();


};
