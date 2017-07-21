#include "manageObjectCnn.h"

extern  std::string pathCaffemodelCnnJSONFile;
extern  std::string pathProtofileCnnJSONFile;

extern  std::string pathCaffemodelSslCnnJSONFile;
extern  std::string pathProtofileSslCnnJSONFile;
extern  std::string pathSolverfileSslCnnJSONFile;
extern 	bool janivanecky;

manageObjectCnn::manageObjectCnn(std::string typeOfNet){

	this->setTypeOfNet(typeOfNet);

	if(this->typeOfNet == CNN_){
		this->setPathToProtoFile();
		this->checkFileExists(this->path2ProtoFile); 
	}

	else{
		this->setPathToSolverFile();
		this->checkFileExists(this->path2SolverFile);
	}

	this->setPathToCaffemodel();
	this->checkFileExists(this->path2Caffemodel);  

	this->createCnn();  
	this->copyWeights2Cnn();
	this->setImageInputLayer();
	this->setOutputLayer();
	this->setNumberOfInputChannels();
	this->setSizeInputLayer();
	this->setSizeOutputLayer();
	this->setPointerToCnnInputData();
	this->setPointerToCnnOutputData();
	this->setPointerToGroundTruthInputData();
	this->setPointerToSparseMatrixData();			
	this->allocateCnnDepthMap();
	this->setScaleDepthMap(1.0);

}

manageObjectCnn::~manageObjectCnn(){}

void manageObjectCnn::setTypeOfNet(std::string typeOfNet){

		if(strcmp("solver", typeOfNet.c_str() ) == 0)
			this->typeOfNet = SOLVER_;

		else
			this->typeOfNet = CNN_;	

}

void manageObjectCnn::saveSnapshot(){

	this->solver->Snapshot();

}

void manageObjectCnn::forwardPassCnn(bool learn){

	if(this->typeOfNet == SOLVER_){
		if(learn)
			this->solver->Step(1);

		else
			this->solver->net()->Forward();
	}

	else
		this->cnn->Forward();

}

void manageObjectCnn::setPathToProtoFile(){

	if(this->typeOfNet == SOLVER_)
		this->path2ProtoFile = pathProtofileSslCnnJSONFile;

	else
		this->path2ProtoFile = pathProtofileCnnJSONFile;

}

void manageObjectCnn::setPathToSolverFile(){
	this->path2SolverFile = pathSolverfileSslCnnJSONFile;
}

void manageObjectCnn::setPathToCaffemodel(){

	if(this->typeOfNet == SOLVER_)
		this->path2Caffemodel = pathCaffemodelSslCnnJSONFile;
	
	else
		this->path2Caffemodel = pathCaffemodelCnnJSONFile;

}

void manageObjectCnn::setOutputLayer(){

	if(this->typeOfNet == SOLVER_){
		this->blobOutputLayer = this->solver->net()->blob_by_name("cnnDepth"); 
		this->blobGroundTruthLayer = this->solver->net()->blob_by_name("groundTruthData");
	}

	else{
		this->blobOutputLayer = this->cnn->blob_by_name("cnnDepth"); 
		this->blobGroundTruthLayer = this->cnn->blob_by_name("groundTruthData");
	}

}

void manageObjectCnn::setImageInputLayer(){

	if(this->typeOfNet == SOLVER_){
		this->blobImageInputLayer =  this->solver->net()->blob_by_name("inputData");
	}

	else{
		this->blobImageInputLayer = this->cnn->blob_by_name("inputData");
	}

}

void manageObjectCnn::setOutputLayer(std::string outputLayerNamer){

	if(this->typeOfNet == SOLVER_)
		this->blobOutputLayer = this->solver->net()->blob_by_name(outputLayerNamer); 

	else
		this->blobOutputLayer = this->cnn->blob_by_name(outputLayerNamer); 


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

	if(this->typeOfNet == SOLVER_){
		caffe::ReadSolverParamsFromTextFileOrDie(this->path2SolverFile, &(this->solver_param));
	    boost::shared_ptr<caffe::Solver<float> > solverr(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	    this->solver = solverr;
	}

	else
		this->cnn.reset(new caffe::Net<float>(this->path2ProtoFile, caffe::TEST) );
	
}

void manageObjectCnn::copyWeights2Cnn(){

	if(!this->path2Caffemodel.empty()){
		if(this->typeOfNet == SOLVER_)
			this->solver->net()->CopyTrainedLayersFrom(this->path2Caffemodel);

		else
			this->cnn->CopyTrainedLayersFrom(this->path2Caffemodel);
	}

}


void manageObjectCnn::setSizeInputLayer(){

	if(this->typeOfNet == SOLVER_){
		this->inputLayerSize.height = this->blobImageInputLayer->shape(2);
		this->inputLayerSize.width  = this->blobImageInputLayer->shape(3);
	}

	else{

		this->inputLayerSize.height = this->blobImageInputLayer->shape(2);
		this->inputLayerSize.width  = this->blobImageInputLayer->shape(3);
	}

}

cv::Size manageObjectCnn::getSizeInputLayer(){

	return( this->inputLayerSize);

}

void manageObjectCnn::setSizeOutputLayer(){

	this->outputLayerSize.height = (this->blobOutputLayer)->shape(2);
	this->outputLayerSize.width  = (this->blobOutputLayer)->shape(3);

}

cv::Size manageObjectCnn::getSizeOutputLayer(){

	return( this-> outputLayerSize );

}

void manageObjectCnn::copyInputMap2InputLayer( cv::Mat inputMap ){

	std::vector<cv::Mat> inputMapInSeparateChannels;
	inputMap.convertTo(inputMap, CV_32FC3);

	for (int currentChannel = 0 ; currentChannel < 	this->numberChannelInputImage; ++currentChannel) {
        cv::Mat channel(this->inputLayerSize.height,this->inputLayerSize.width, CV_32FC1, this->pointerToCnnInputMap );
        inputMapInSeparateChannels.push_back(channel);
        this->pointerToCnnInputMap += this->inputLayerSize.width * this->inputLayerSize.height;
    }

	cv::split(inputMap, inputMapInSeparateChannels);

	if(this->typeOfNet == SOLVER_)   
		CHECK(reinterpret_cast<float*>(inputMapInSeparateChannels.at(0).data)  == (this->solver)->net()->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";

	else
		CHECK(reinterpret_cast<float*>(inputMapInSeparateChannels.at(0).data)  == (this->cnn)->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";

	this->setPointerToCnnInputData();
}


void manageObjectCnn::copyGroundTruthInputMap2GroundTruthInputLayer( cv::Mat inputMap ){

	std::vector<cv::Mat> inputMapInSeparateChannels;
	int numberChannelInputImage = (this->blobGroundTruthLayer)->shape(1);
	inputMap.convertTo(inputMap, CV_32FC1);

	for (int currentChannel2 = 0 ; currentChannel2 < numberChannelInputImage ; ++currentChannel2) {
        cv::Mat channel2(this->outputLayerSize.height,this->outputLayerSize.width, CV_32FC1, this->pointerToGroundTruthInputMap );
        inputMapInSeparateChannels.push_back(channel2);
        this->pointerToGroundTruthInputMap += this->outputLayerSize.width * this->outputLayerSize.height;
    }

	cv::split(inputMap, inputMapInSeparateChannels);
    CHECK(reinterpret_cast<float*>(inputMapInSeparateChannels.at(0).data)  == (this->solver)->net()->blob_by_name("groundTruthData")->cpu_data() ) << "Input channels are not wrapping the input layer of the network.";

	this->setPointerToGroundTruthInputData();

}

void manageObjectCnn::copySparseLayer( cv::Mat inputMap ){

	std::vector<cv::Mat> inputMapInSeparateChannels;
	int numberChannelInputImage = 1;
	inputMap.convertTo(inputMap, CV_32FC1);
	cv::resize(inputMap,inputMap,cv::Size(this->outputLayerSize.width , this->outputLayerSize.height));

	for (int currentChannel2 = 0 ; currentChannel2 < numberChannelInputImage ; ++currentChannel2) {
        cv::Mat channel2(this->outputLayerSize.height,this->outputLayerSize.width, CV_32FC1, this->pointerToSparse );
        inputMapInSeparateChannels.push_back(channel2);
        this->pointerToSparse += this->outputLayerSize.width * this->outputLayerSize.height;
    }

	cv::split(inputMap, inputMapInSeparateChannels);
    CHECK(reinterpret_cast<float*>(inputMapInSeparateChannels.at(0).data)  == (this->solver)->net()->blob_by_name("sparseData")->cpu_data() ) << "Input channels are not wrapping the input layer of the network.";

	this->setPointerToSparseMatrixData();

}

	cv::Mat manageObjectCnn::getCnnOutputMap(){

	return(this->cnnDepthMap);

}

void manageObjectCnn::setPointerToCnnOutputData(){

	if(this->typeOfNet == SOLVER_)
		this->pointerToCnnOutputMap = (this->solver)->net()->blob_by_name("cnnDepth")->mutable_cpu_data();

	else
		this->pointerToCnnOutputMap = (this->cnn)->blob_by_name("cnnDepth")->mutable_cpu_data();

}
void manageObjectCnn::setPointerToSparseMatrixData(){
	if(this->typeOfNet == SOLVER_){
		this->pointerToSparse = (this->solver)->net()->blob_by_name("sparseData")->mutable_cpu_data();
	}
}

void manageObjectCnn::setPointerToCnnInputData(){

		this->pointerToCnnInputMap = this->blobImageInputLayer->mutable_cpu_data();
}

void manageObjectCnn::setPointerToGroundTruthInputData(){

	if(this->typeOfNet == SOLVER_){
		this->pointerToGroundTruthInputMap =  (this->solver)->net()->blob_by_name("groundTruthData")->mutable_cpu_data();
	}

	else{
		this->pointerToGroundTruthInputMap =  (this->cnn)->blob_by_name("groundTruthData")->mutable_cpu_data();
	}

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

void manageObjectCnn::setNumberOfInputChannels(){

	if(this->typeOfNet == SOLVER_){
			this->numberChannelInputImage = ((this->solver)->net()->input_blobs()[0])->shape(1);
	}

	else{
			this->numberChannelInputImage = ((this->cnn)->input_blobs()[0])->shape(1);
	}

}

void manageObjectCnn::setScaleDepthMap(float scale){

	this->depthScale = scale;

}

void manageObjectCnn::computeMeanDepthMap(){

	int counterPositiveSamples=0;
	float sumOfDepths = 0.0;

	for(int currentRow = 0; currentRow < this->outputLayerSize.height;currentRow++){

		for(int currentCol =0;currentCol<this->outputLayerSize.width;currentCol++){

			if(this->cnnDepthMap.at<float>(currentRow,currentCol) > 0.0){
				counterPositiveSamples++;
				sumOfDepths = sumOfDepths + this->cnnDepthMap.at<float>(currentRow,currentCol);
			}
		}
	}
	this->meanDepthMap = sumOfDepths/counterPositiveSamples*this->depthScale;

}

void manageObjectCnn::replaceNegativeDepths(){
	int inv = 0;
	for(int currentRow = 0; currentRow < this->outputLayerSize.height;currentRow++){
		for(int currentCol =0;currentCol<this->outputLayerSize.width;currentCol++){

			if(this->cnnDepthMap.at<float>(currentRow,currentCol) < 0.0 ){
				this->cnnDepthMap.at<float>(currentRow,currentCol) = 0.0;
				inv++;
			}

			 if(this->cnnDepthMap.at<float>(currentRow,currentCol) > 1.0 && !janivanecky ){
				this->cnnDepthMap.at<float>(currentRow,currentCol) = 1.0;
				inv++;
			}
				
		}
	}
}
