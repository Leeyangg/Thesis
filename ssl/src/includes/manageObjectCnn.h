//Caffe includes
#include <caffe/caffe.hpp>

// C includes
#include <memory>
#include <stdio.h>

//Opencv includes
#include "opencv2/opencv.hpp"

class manageObjectCnn
{
public:
	manageObjectCnn();
	~manageObjectCnn();
	cv::Size getSizeInputLayer();
	cv::Size getSizeOutputLayer();
	void copyInputMap2InputLayer(cv::Mat inputMap);
	void forwardPassCnn();
	cv::Mat getCnnOutputMap();
	void extractDepthMapCnn();


private:

	std::string path2ProtoFile;
	std::string path2Caffemodel;
	std::shared_ptr<caffe::Net<float> > cnn;
	cv::Size inputLayerSize;
	cv::Size outputLayerSize;
	const float* pointerToCnnOutputMap;
	float* pointerToCnnInputMap;
	cv::Mat cnnDepthMap;
	void setParametersCnn();
	void setPathToProtoFile();
	void setPathToCaffemodel();
	void setPointerToCnnOutputData();
	void setPointerToCnnInputData();
	bool checkFileExists(std::string path2File);
	void createCnn();
	void copyWeights2Cnn();
	void setSizeInputLayer();
	void setSizeOutputLayer();
	void allocateCnnDepthMap();

};

