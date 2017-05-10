// C includes
#include <string>
#include <iostream>

// Opencv includes
#include "opencv2/opencv.hpp"

class manageObjectInputMap
{
public:
	manageObjectInputMap(std::string type, cv::Size desiredInputSize);
	~manageObjectInputMap();
	cv::Mat getInputMap();
	cv::Mat getInputMapResized();
	void setPath2SourceFolderInputMap();
	void setInputMapFormat();
	void setNumberFirstInputMap();
	bool testInputMapExists();
	void readInputMap();
	std::string getPath2InputMap();
	void displayInputMap();
	void displayInputMapResized();
	void setPath2InputMap();
	void updatePath2InputMap();
	void setSizeInputMap(cv::Size desiredInputSize);
	void resizeInputMap();
	
protected:


private:
	cv::Mat inputMap;
	std::string path2SourceFolderInputMap;
	std::string inputMapFormat;
	int numberCurrentInputMap;
	bool inputMapExists; 
	std::string path2InputMap;
	std::string inputMapType;
	cv::Size desiredSizeInputMap;
	cv::Mat inputMapResized;
	void createInputMatrixResized();
	void setInputMapType(std::string inputMapType); 
};