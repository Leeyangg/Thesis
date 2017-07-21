// C includes
#include <string>
#include <iostream>

//My include
#include "manageZEDObject.h"

// Opencv includes
#include "opencv2/opencv.hpp"

#define DEPTH_MAP_ 			0 
#define IMAGE_MAP_ 			1
#define CONFIDENCE_MAP_ 	2
#define ZED_CONFIDENCE_MAP_ 3
#define ZED_CAM_MAP_		4
#define ZED_DEPTH_MAP_ 		5 
#define STEREO_IMAGE_ 		6 
#define CHEAP_DEPTH_ 		7 

class manageObjectInputMap
{
public:
	manageObjectInputMap(std::string type, cv::Size desiredInputSize);
    manageObjectInputMap(std::string type, cv::Size desiredInputSize, manageZEDObject* zedCamObject);
	~manageObjectInputMap();
	cv::Mat getInputMap();
	cv::Mat getRightMap();
	cv::Mat getInputMapResized();
	cv::Mat getRightMapResized();
	void setPath2SourceInputMap();
	void setInputMapFormat();
	void setNumberFirstInputMap();
	bool testInputMapExists();
	void readInputMap();
	std::string getPath2InputMap();
	void displayInputMap();
	void displayInputMapResized();
	void setPath2InputMap();
	void updateInputMap();
	void setSizeInputMap(cv::Size desiredInputSize);
	void resizeInputMap();
	
protected:

private:


	std::string path2SourceFolderInputMap;
	std::string path2SourceFolderRightMap;
	std::string inputMapFormat;
	std::string path2InputMap;
	std::string path2RightMap;
	bool inputMapExists; 
	int numberCurrentInputMap;
	int inputMapType;
	cv::Size desiredSizeInputMap;
	cv::Mat inputMapResized;
	cv::Mat rightMapResized;
	cv::Mat inputMap;
	cv::Mat rightMap;
	manageZEDObject* zedCamObject;
	void createInputMatrixResized();
	void setInputMapType(std::string inputMapType); 
};
