      #include "ssl.h"

void grabFrameZed(manageZEDObject* zedCamObject){

	for(;;){
		zedCamObject->grabFrame();
	}
}

void setupStart(int choice){

	solver = new manageObjectCnn("solver");
	cnn = new manageObjectCnn("cnn");

	if(choice == NYUDataset){
		scaleInputDepthMap = 6.0/255.0;
		scaleOriginalCnnMap = 10.0;
		scaleSSLCnnMap= 6.0;
		mergeFromConfidenceMap = false;
		inputImage = new manageObjectInputMap("image", solver->getSizeInputLayer());
		inputDepthMap = new manageObjectInputMap("depth", solver->getSizeOutputLayer());
		inputConfidenceMap = new manageObjectInputMap("confidence", solver->getSizeOutputLayer());
	}

	else if(choice == MyDataset){
		scaleInputDepthMap = 10.0/255.0;
		scaleOriginalCnnMap = 10.0;
		scaleSSLCnnMap =10.0;
		mergeFromConfidenceMap = true;
		inputImage = new manageObjectInputMap("zed", solver->getSizeInputLayer(), zedCamObject);
		inputDepthMap = new manageObjectInputMap("zed_depth", solver->getSizeOutputLayer(),zedCamObject);
		inputConfidenceMap = new manageObjectInputMap("zed_confidence", solver->getSizeOutputLayer(),zedCamObject);
	}

	else if(choice == ZED){
		scaleInputDepthMap = 10.0/255.0;
		scaleOriginalCnnMap = 10.0;
		scaleSSLCnnMap =10.0;
		mergeFromConfidenceMap = true;
		zedCamObject = new manageZEDObject;
		inputImage = new manageObjectInputMap("zed", solver->getSizeInputLayer(), zedCamObject);
		inputDepthMap = new manageObjectInputMap("zed_depth", solver->getSizeOutputLayer(),zedCamObject);
		inputConfidenceMap = new manageObjectInputMap("zed_confidence", solver->getSizeOutputLayer(),zedCamObject);
		std::thread release(grabFrameZed, zedCamObject);
	}

	thresholdConfidence = 30;

}

int  main(int argc, char const *argv[])
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	int activeWindow = 1;
	bool quit = false;
	std::ofstream costsFile;
	costsFile.open("costs.txt");
	
	if(argc == 2)
		setupStart(atoi(argv[1]));

	else{
		std::cout << "Please image source" << std::endl;
		return(0);
	}

	for(;;){

		inputImage->readInputMap();
		inputImage->resizeInputMap();


		if(mergeFromConfidenceMap){
			inputConfidenceMap->readInputMap();
			inputConfidenceMap->resizeInputMap();
		}

		inputDepthMap->readInputMap();
		inputDepthMap->resizeInputMap();	


		cv::Mat noiseMatrix(solver->getSizeOutputLayer(), CV_32FC1);
		cv::Mat inputMm(solver->getSizeOutputLayer(), CV_32FC1);
		cv::randn(noiseMatrix,0,10);
		inputDepthMap->getInputMapResized().convertTo(inputMm,CV_32FC1);
		//noiseMatrix = noiseMatrix.mul(inputMm);
		cv::add(noiseMatrix,inputMm,noiseMatrix);
		

		solver->copyInputMap2InputLayer(inputImage->getInputMapResized());
		//solver->copyGroundTruthInputMap2GroundTruthInputLayer(inputDepthMap->getInputMapResized());
		solver->copyGroundTruthInputMap2GroundTruthInputLayer(inputDepthMap->getInputMapResized());
		solver->forwardPassCnn();
		solver->extractDepthMapCnn();
		solver->setScaleDepthMap(scaleSSLCnnMap);
		solver->computeMeanDepthMap();
		solver->replaceNegativeDepths();

     	cnn->copyInputMap2InputLayer(inputImage->getInputMapResized());
		cnn->copyGroundTruthInputMap2GroundTruthInputLayer(inputDepthMap->getInputMapResized());
	    cnn->forwardPassCnn();
		cnn->extractDepthMapCnn();
		cnn->setScaleDepthMap(scaleOriginalCnnMap);
		cnn->computeMeanDepthMap();
		cnn->replaceNegativeDepths();

	    depthCnn.setDepthMap(solver->getCnnOutputMap());
	    depthCnn.setThresholdFilter(thresholdConfidence);

	    if(mergeFromConfidenceMap)
			depthCnn.filterPixels2BeMerged(inputConfidenceMap->getInputMapResized());

		else
			depthCnn.filterPixels2BeMerged();

		depthCnn.mergeDepthMap(inputDepthMap->getInputMapResized(), "facil", scaleInputDepthMap, scaleSSLCnnMap);
		depthCnn.refreshPixels2BeMerged();

		performanOriginalCnnMap = new manageDepthMapPerformance;
		performanOriginalCnnMap->setDepthMapGroundTruth(inputDepthMap->getInputMapResized());
		performanOriginalCnnMap->setDepthMapEstimation(cnn->getCnnOutputMap());
		performanOriginalCnnMap->setScaleDepthMap(scaleOriginalCnnMap);
		performanOriginalCnnMap->setScaleGroundTruth(scaleInputDepthMap);
		performanOriginalCnnMap->computePerformance();
		//std::cout << "Error original CNN: " <<  performanOriginalCnnMap->getLinearRMSE() << std::endl;

		performanCnnMap = new manageDepthMapPerformance;
		performanCnnMap->setDepthMapGroundTruth(inputDepthMap->getInputMapResized());
		performanCnnMap->setDepthMapEstimation(solver->getCnnOutputMap());
		performanCnnMap->setScaleDepthMap(scaleSSLCnnMap);
		performanCnnMap->setScaleGroundTruth(scaleInputDepthMap);
		performanCnnMap->computePerformance();
		//std::cout << "Error SSL CNN: " << performanCnnMap->getLinearRMSE() << std::endl;

		performanceMergedMap = new manageDepthMapPerformance;
		performanceMergedMap->setDepthMapGroundTruth(inputDepthMap->getInputMapResized());
		performanceMergedMap->setDepthMapEstimation(depthCnn.getMergedDepthMap());
		performanceMergedMap->setScaleDepthMap(1.0);
		performanceMergedMap->setScaleGroundTruth(scaleInputDepthMap);
		performanceMergedMap->computePerformance();
		//std::cout << "Error merged map " << performanceMergedMap->getLinearRMSE() << std::endl;

		costsFile << performanOriginalCnnMap->getLinearRMSE() << " "  << performanCnnMap->getLinearRMSE()  << " " << performanceMergedMap->getLinearRMSE() << std::endl;


		displayDepthColorMap.setMap(inputDepthMap->getInputMap(), "Input Stereo Depth Map");	
		displayDepthColorMap.setScaleFactor(1.0);
		displayDepthColorMap.useColorMap(1);
		displayDepthColorMap.displayColorMat();
	    displayDepthCnnColorMap.setMap(solver->getCnnOutputMap(), "SSL CNN Depth Map");
		displayDepthCnnColorMap.setScaleFactor(255.0);
		displayDepthCnnColorMap.useColorMap(1);
		displayDepthCnnColorMap.displayColorMat();
		displayDepthOriginalCnnColorMap.setMap(cnn->getCnnOutputMap(), "Original CNN Depth Map");
		displayDepthOriginalCnnColorMap.setScaleFactor(255.0);
		displayDepthOriginalCnnColorMap.useColorMap(1);
		displayDepthOriginalCnnColorMap.displayColorMat();
		displayDepthOriginalCnnColorMap.setMap(depthCnn.getMergedDepthMap(), "Merged Depth Map");
		displayDepthOriginalCnnColorMap.setScaleFactor(255.0/scaleSSLCnnMap);
		displayDepthOriginalCnnColorMap.useColorMap(1);
		displayDepthOriginalCnnColorMap.displayColorMat();
		displayNoisyDepthMap.setMap(noiseMatrix, "Noisy Depth Map");
		displayNoisyDepthMap.setScaleFactor(1);
		displayNoisyDepthMap.useColorMap(1);
		displayNoisyDepthMap.displayColorMat();
		inputImage->displayInputMapResized();

		cv::waitKey(30);

		inputImage->updateInputMap();
		inputDepthMap->updateInputMap();

		if(mergeFromConfidenceMap)
			inputConfidenceMap->updateInputMap();

	}

	std::cout << "Leaving SSL..." << std::endl;
	costsFile.close();

	return 0;

}




