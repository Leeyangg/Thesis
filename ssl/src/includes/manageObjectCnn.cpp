#include "manageObjectCnn.h"

manageObjectCnn::manageObjectCnn(){

	this->setPathToProtoFile();
	this->checkFileExists(path2ProtoFile);
	this->setPathToCaffemodel();
	this->checkFileExists(path2Caffemodel);
	this->createCnn();
	this->copyWeights2Cnn();
	this->setSizeInputLayer();
	this->setSizeOutputLayer();
	this->setPointerToCnnInputData();
	this->setPointerToCnnOutputData();
	this->allocateCnnDepthMap();
	
}

manageObjectCnn::~manageObjectCnn(){}

void manageObjectCnn::setPathToProtoFile(){

	std::cout << "Insert path to protofile:" << std::endl;
	this->path2ProtoFile = "/home/diogo/Desktop/Thesis/display_cnn_stereo/mix_eigen/mix_eigen_deploy.prototxt";
	//std::cin.sync();
	//std::cin >> this->path2ProtoFile;
}

void manageObjectCnn::setPathToCaffemodel(){

	std::cout << "Insert path to Caffemodel:"<< std::endl;
	this->path2Caffemodel = "/home/diogo/Desktop/Thesis/display_cnn_stereo/mix_eigen/m.caffemodel";
	//std::cin.sync();
	//std::cin >> this->path2Caffemodel;

}

bool manageObjectCnn::checkFileExists(std::string path2File){

	FILE *testFilePointer;

	if(testFilePointer = fopen(path2File.c_str(), "r")){
		fclose(testFilePointer);
		return true;
	}

	else{
		std::cout << path2File << " Not found. Leaving..." << std::endl;
		return false;
	}

}

void manageObjectCnn::createCnn(){

	this->cnn.reset(new caffe::Net<float>(this->path2ProtoFile, caffe::TEST));

}

void manageObjectCnn::copyWeights2Cnn(){

	cnn->CopyTrainedLayersFrom(this->path2Caffemodel);

}


void manageObjectCnn::setSizeInputLayer(){

	this->inputLayerSize.height = ((this->cnn)->input_blobs()[0])->shape(2);
	this->inputLayerSize.width  = ((this->cnn)->input_blobs()[0])->shape(3);

}

cv::Size manageObjectCnn::getSizeInputLayer(){

	return( this->inputLayerSize);

}

void manageObjectCnn::setSizeOutputLayer(){

	this->outputLayerSize.height = ((this->cnn)->output_blobs()[0])->shape(2);
	this->outputLayerSize.width  = ((this->cnn)->output_blobs()[0])->shape(3);

}

cv::Size manageObjectCnn::getSizeOutputLayer(){

	return( this-> outputLayerSize );

}

void manageObjectCnn::copyInputMap2InputLayer( cv::Mat inputMap ){

	std::vector<cv::Mat> inputMapInSeparateChannels;
	int numberChannelInputImage = ((this->cnn)->input_blobs()[0])->shape(1);
	inputMap.convertTo(inputMap, CV_32FC3);

    for (int currentChannel = 0 ; currentChannel < numberChannelInputImage ; ++currentChannel) {
        cv::Mat channel(this->inputLayerSize.height,this->inputLayerSize.width, CV_32FC1, this->pointerToCnnInputMap);
        inputMapInSeparateChannels.push_back(channel);
        this->pointerToCnnInputMap += this->inputLayerSize.width * this->inputLayerSize.height;
    }

	cv::split(inputMap, inputMapInSeparateChannels);
    CHECK(reinterpret_cast<float*>(inputMapInSeparateChannels.at(0).data)  == ((this->cnn)->input_blobs()[0])->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
	this->setPointerToCnnInputData();

}

void manageObjectCnn::forwardPassCnn(){

	(this->cnn)->Forward();

}

cv::Mat manageObjectCnn::getCnnOutputMap(){

	return(this->cnnDepthMap);

}

void manageObjectCnn::setPointerToCnnOutputData(){

	this->pointerToCnnOutputMap = ((this->cnn)->output_blobs()[0])->cpu_data();
	//this->pointerToCnnOutputMap = (this->cnn)->blob_by_name("fine_depth")->cpu_data();
}

void manageObjectCnn::setPointerToCnnInputData(){

	this->pointerToCnnInputMap = ((this->cnn)->input_blobs()[0])->mutable_cpu_data();

}

void manageObjectCnn::allocateCnnDepthMap(){

	this->cnnDepthMap.create(this->outputLayerSize.height, this->outputLayerSize.width, CV_32FC1);

}

void manageObjectCnn::extractDepthMapCnn(){

	float arrayCnnOutput[this->outputLayerSize.width*this->outputLayerSize.height*sizeof(float)];
	memcpy( &arrayCnnOutput[0], (float*) this->pointerToCnnOutputMap, this->outputLayerSize.width*this->outputLayerSize.height*sizeof(float));

	float* currentPointerToMemoryDestination;
	float* currentPointerToMemorySource = &arrayCnnOutput[0] ;

	for(int currentRowMatrixDepth = 0; currentRowMatrixDepth <  this->outputLayerSize.height; currentRowMatrixDepth++){
		
		currentPointerToMemoryDestination = (float*) this->cnnDepthMap.ptr(currentRowMatrixDepth);
		memcpy( currentPointerToMemoryDestination, currentPointerToMemorySource, this->outputLayerSize.width*sizeof(float));
		currentPointerToMemorySource = currentPointerToMemorySource + this->outputLayerSize.width;

	}
}