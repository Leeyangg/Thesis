//Caffe includes
#include <caffe/caffe.hpp>

// C includes
#include <memory>
#include <stdio.h>

//Opencv includes
#include "opencv2/opencv.hpp"

//My defines
#define CNN_ 1
#define SOLVER_ 0 


class manageObjectCnn
{
public:
	manageObjectCnn(std::string typeOfNet);
	~manageObjectCnn();
	cv::Size getSizeInputLayer();
	cv::Size getSizeOutputLayer();
	void copyInputMap2InputLayer(cv::Mat inputMap);
	void copyGroundTruthInputMap2GroundTruthInputLayer(cv::Mat inputMap);
	void forwardPassCnn();
	cv::Mat getCnnOutputMap();
	void extractDepthMapCnn();
	void setOutputLayer(std::string outputLayerNamer);

private:

	std::string path2ProtoFile;
	std::string path2SolverFile;
	std::string path2Caffemodel;
	std::shared_ptr<caffe::Net<float> > cnn;
	cv::Size inputLayerSize;
	cv::Size outputLayerSize;
	const float* pointerToCnnOutputMap;
	float* pointerToCnnInputMap;
	float* pointerToSolverCnnInputMap;
	float* pointerToGroundTruthInputMap;
	float* pointerToSolverGroundTruthInputMap;
	cv::Mat cnnDepthMap;
	boost::shared_ptr<caffe::Blob<float>> blobImageInputLayer;
	boost::shared_ptr<caffe::Blob<float>> blobOutputLayer;
	boost::shared_ptr<caffe::Blob<float>> blobGroundTruthLayer;
	int typeOfNet;
	int numberChannelInputImage;
	void setParametersCnn();
	void setPathToProtoFile();
	void setPathToSolverFile();
	void setPathToCaffemodel();
	void setPointerToCnnOutputData();
	void setPointerToCnnInputData();
	bool checkFileExists(std::string path2File);
	void createCnn();
	void copyWeights2Cnn();
	void setSizeInputLayer();
	void setSizeOutputLayer();
	void allocateCnnDepthMap();
	void setPointerToGroundTruthInputData();
	void setTypeOfNet(std::string typeOfNet);
	void setNumberOfInputChannels();
	void setOutputLayer();
	void setImageInputLayer();
    caffe::SolverParameter solver_param;
    boost::shared_ptr<caffe::Solver<float> > solver;	 

};

