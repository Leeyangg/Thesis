#include "ssl.h"

void grabFrameZed(manageZEDObject* zedCamObject){

	for(;;){
		zedCamObject->grabFrame();
		cv::imshow("Input Image",zedCamObject->getImage() );
		cv::waitKey(5);
	}
}

void setupStart(int choice){

	if(choice == NYUDataset){
		scaleInputDepthMap = 6.0/255.0;
		scaleOriginalCnnMap = 10.0;
		scaleSSLCnnMap= 6.0;
		mergeFromConfidenceMap = false;
	}

	else if(choice == MyDataset){
		scaleInputDepthMap = 10.0/255.0;
		scaleOriginalCnnMap = 10.0;
		scaleSSLCnnMap =10.0;
		mergeFromConfidenceMap = true;

	}
}


int  main(int argc, char const *argv[])
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	int activeWindow = 1;
	bool quit = false;
	std::ofstream costsFile;
	costsFile.open("costs.txt");
	//costsFile << "Original_CNN  SSL_CNN" << std::endl;

	//setupStart(MyDataset);
	setupStart(NYUDataset);

	manageObjectCnn solver("solver");
	manageObjectCnn cnn("cnn");
    manageObjectInputMap inputImage("image", solver.getSizeInputLayer());
	manageObjectInputMap inputDepthMap("depth", solver.getSizeOutputLayer());
	manageObjectInputMap inputConfidenceMap("confidence", solver.getSizeOutputLayer());
	manageObjectDepthMap depthCnn;
	manageDepthMapPerformance * performanOriginalCnnMap;
	manageDepthMapPerformance * performanCnnMap;
	manageDepthMapPerformance * performanceMergedMap;
	displayObjectDepthMap displayDepthColorMap;
	displayObjectDepthMap displayDepthCnnColorMap;
	displayObjectDepthMap displayDepthOriginalCnnColorMap;

	for(;;){

		inputImage.readInputMap();
		inputImage.resizeInputMap();
		
		if(mergeFromConfidenceMap){
			inputConfidenceMap.readInputMap();
			inputConfidenceMap.resizeInputMap();
		}

		inputDepthMap.readInputMap();
		inputDepthMap.resizeInputMap();	

		solver.copyInputMap2InputLayer(inputImage.getInputMapResized());
		solver.copyGroundTruthInputMap2GroundTruthInputLayer(inputDepthMap.getInputMapResized());
		solver.forwardPassCnn();
		solver.extractDepthMapCnn();
		solver.setScaleDepthMap(scaleSSLCnnMap);
		solver.computeMeanDepthMap();
		solver.replaceNegativeDepths();

     	cnn.copyInputMap2InputLayer(inputImage.getInputMapResized());
		cnn.copyGroundTruthInputMap2GroundTruthInputLayer(inputDepthMap.getInputMapResized());
	    cnn.forwardPassCnn();
		cnn.extractDepthMapCnn();
		cnn.setScaleDepthMap(scaleOriginalCnnMap);
		cnn.computeMeanDepthMap();
		cnn.replaceNegativeDepths();

	    depthCnn.setDepthMap(solver.getCnnOutputMap());
	    depthCnn.setThresholdFilter(30);

	    if(mergeFromConfidenceMap)
			depthCnn.filterPixels2BeMerged(inputConfidenceMap.getInputMapResized());

		else
			depthCnn.filterPixels2BeMerged();

		depthCnn.mergeDepthMap(inputDepthMap.getInputMapResized(), "facil", scaleInputDepthMap, scaleSSLCnnMap);
		depthCnn.refreshPixels2BeMerged();

		performanOriginalCnnMap = new manageDepthMapPerformance;
		performanOriginalCnnMap->setDepthMapGroundTruth(inputDepthMap.getInputMapResized());
		performanOriginalCnnMap->setDepthMapEstimation(cnn.getCnnOutputMap());
		performanOriginalCnnMap->setScaleDepthMap(scaleOriginalCnnMap);
		performanOriginalCnnMap->setScaleGroundTruth(scaleInputDepthMap);
		performanOriginalCnnMap->computePerformance();
		std::cout << "Error original CNN: " <<  performanOriginalCnnMap->getLinearRMSE() << std::endl;

		performanCnnMap = new manageDepthMapPerformance;
		performanCnnMap->setDepthMapGroundTruth(inputDepthMap.getInputMapResized());
		performanCnnMap->setDepthMapEstimation(solver.getCnnOutputMap());
		performanCnnMap->setScaleDepthMap(scaleSSLCnnMap);
		performanCnnMap->setScaleGroundTruth(scaleInputDepthMap);
		performanCnnMap->computePerformance();
		std::cout << "Error SSL CNN: " << performanCnnMap->getLinearRMSE() << std::endl;

		performanceMergedMap = new manageDepthMapPerformance;
		performanceMergedMap->setDepthMapGroundTruth(inputDepthMap.getInputMapResized());
		performanceMergedMap->setDepthMapEstimation(depthCnn.getMergedDepthMap());
		performanceMergedMap->setScaleDepthMap(1.0);
		performanceMergedMap->setScaleGroundTruth(scaleInputDepthMap);
		performanceMergedMap->computePerformance();
		std::cout << "Error merged map " << performanceMergedMap->getLinearRMSE() << std::endl;

		costsFile << performanOriginalCnnMap->getLinearRMSE() << " "  << performanCnnMap->getLinearRMSE()  << " " << performanceMergedMap->getLinearRMSE() << std::endl;


		displayDepthColorMap.setMap(inputDepthMap.getInputMap(), "Input Stereo Depth Map");				
		displayDepthColorMap.setScaleFactor(1.0);
		displayDepthColorMap.useColorMap(1);
		displayDepthColorMap.displayColorMat();
	    displayDepthCnnColorMap.setMap(solver.getCnnOutputMap(), "SSL CNN Depth Map");
		displayDepthCnnColorMap.setScaleFactor(255.0);
		displayDepthCnnColorMap.useColorMap(1);
		displayDepthCnnColorMap.displayColorMat();
		displayDepthOriginalCnnColorMap.setMap(cnn.getCnnOutputMap(), "Original CNN Depth Map");
		displayDepthOriginalCnnColorMap.setScaleFactor(255.0);
		displayDepthOriginalCnnColorMap.useColorMap(1);
		displayDepthOriginalCnnColorMap.displayColorMat();
		displayDepthOriginalCnnColorMap.setMap(depthCnn.getMergedDepthMap(), "Merged Depth Map");
		displayDepthOriginalCnnColorMap.setScaleFactor(255.0/scaleSSLCnnMap);
		displayDepthOriginalCnnColorMap.useColorMap(1);
		displayDepthOriginalCnnColorMap.displayColorMat();
		inputImage.displayInputMapResized();

		cv::waitKey(30);

		inputImage.updateInputMap();
		inputDepthMap.updateInputMap();

		if(mergeFromConfidenceMap)
			inputConfidenceMap.updateInputMap();

	}

	std::cout << "Leaving SSL..." << std::endl;
	costsFile.close();

	return 0;

}




