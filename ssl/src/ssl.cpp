#include "ssl.h"

void setupStart(){

	scaleInputDepthMap = scaleDepthMapSslJSONFile/255.0;
	scaleSSLCnnMap= scaleDepthMapSslJSONFile;
	scaleOriginalCnnMap = scaleDepthMapCnnNoUpdateJSONFile;

   if(useZedJSONFile){
		//if(zedSourceSdkJSONFile)
			mergeFromConfidenceMap = true;

		zedCamObject = new manageZEDObject;
		inputImage = new manageObjectInputMap("zed", resolutionInputMapsJSONFile, zedCamObject);
		inputDepthMap = new manageObjectInputMap("zed_depth", resolutionOutputMapsJSONFile,zedCamObject);
		inputConfidenceMap = new manageObjectInputMap("zed_confidence", resolutionOutputMapsJSONFile,zedCamObject);
		thresholdConfidence = 70;
	}

	if(useCnnSslJSONFile)
		solver = new manageObjectCnn("solver");

	if(useCnnNoWeigthUpdateJSONFile)
		cnn = new manageObjectCnn("cnn");
		

  if(useImportFromFolderJSONFile){
		if(useStereoPairJSONFile)
			mergeFromConfidenceMap = true;

		else
			mergeFromConfidenceMap = false;

		if(useStereoPairJSONFile)
			inputImage = new manageObjectInputMap("stereo_pair", resolutionInputMapsJSONFile);
		else
			inputImage = new manageObjectInputMap("image", resolutionInputMapsJSONFile);

		inputDepthMap = new manageObjectInputMap("depth", resolutionOutputMapsJSONFile);

		if(mergeJSONFile)
			inputConfidenceMap = new manageObjectInputMap("confidence", resolutionOutputMapsJSONFile);
	}

}

extern  bool useZedJSONFile;
extern  bool zedSourceOpenCvJSONFile;
extern  bool zedSourceSdkJSONFile;
extern  bool useImportFromFolderJSONFile;
extern  bool useStereoPairJSONFile;
extern  bool useCnnNoWeigthUpdateJSONFile;
extern  bool useCnnSslJSONFile;
extern  float scaleDepthMapSslJSONFile;
extern  float scaleDepthMapCnnNoUpdateJSONFile;
extern  cv::Size resolutionInputMapsJSONFile;
extern  cv::Size resolutionOutputMapsJSONFile;
extern  bool mergeJSONFile;
extern bool stereoOpenCVJSONFile;

int  main(int argc, char const *argv[])
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	int activeWindow = 1;
	bool quit = false;
	std::ofstream costsFile;
	costsFile.open("costs.txt");
	cv::Mat inputImageCnn;
	cv::Mat depthGT;
	cv::Mat depthMapToBeMerged;
	cv::Mat depthStereoOpenCv;
	cv::Mat leftImage;
	cv::Mat rightImage;
	std::vector<cv::Mat> zedImages;
	config::loadVariablesFromJson();
	setupStart();

	for(;;){

		if(useZedJSONFile){
			zedCamObject->grabFrame();

			if(zedSourceOpenCvJSONFile){
				zedCamObject->getLeftImage(false).copyTo(leftImage);
				zedCamObject->getRightImage(false).copyTo(rightImage);
			}

			else{

				zedImages = zedCamObject->getImage();
				zedImages.at(0).copyTo(leftImage);
				zedImages.at(1).copyTo(rightImage);
				zedCamObject->getDepthMap().copyTo(depthGT);
				displayDepthColorMap.setMap(depthGT, "Ground Truth Depth Map");	
				displayDepthColorMap.setScaleFactor(1.0);
				displayDepthColorMap.useColorMap(1);
				displayDepthColorMap.displayColorMat();

			}

			leftImage.copyTo(inputImageCnn);
			cv::imshow("Original left image", leftImage);

//			if(zedSourceSdkJSONFile){


//			}

	//		cv::resize(zedCamObject->getImage(), inputImageCnn, resolutionInputMapsJSONFile);
		}

		if(useImportFromFolderJSONFile){

			inputImage->readInputMap();
			inputImage->resizeInputMap();
			inputImage->getInputMapResized().copyTo(inputImageCnn);
			inputImage->updateInputMap();
/*
			if(mergeFromConfidenceMap){
				inputConfidenceMap->readInputMap();
				inputConfidenceMap->resizeInputMap();
			}

			inputDepthMap->readInputMap();
			inputDepthMap->resizeInputMap();
			inputDepthMap->getInputMapResized().copyTo(depthGT);	
			inputImage->updateInputMap();
			inputDepthMap->updateInputMap();
			inputImage->displayInputMapResized();
			displayDepthColorMap.setMap(inputDepthMap->getInputMap(), "Input Stereo Depth Map");	
			displayDepthColorMap.setScaleFactor(1.0);
			displayDepthColorMap.useColorMap(1);
			displayDepthColorMap.displayColorMat();
*/
			if(useStereoPairJSONFile){
				inputImage->getInputMapResized().copyTo(leftImage);
				inputImage->getRightMapResized().copyTo(rightImage);
			}

		}

		if(stereoOpenCVJSONFile){
			bmAlgorithm.setResolution(resolutionOutputMapsJSONFile);
			bmAlgorithm.setScaleDepthMap(700.262*0.120);

			bmAlgorithm.setLeftImage(leftImage);
			bmAlgorithm.setRightImage(rightImage);
			bmAlgorithm.computeDisparityMap();
			bmAlgorithm.computeAbsoluteDepthMap();
			bmAlgorithm.getAbsoluteDepthMapResized().copyTo(depthStereoOpenCv);
			displayDepthOriginalCnnColorMap.setMap(depthStereoOpenCv, "Depth OpenCVBM");
			displayDepthOriginalCnnColorMap.setScaleFactor(255.0/10.0);
			displayDepthOriginalCnnColorMap.useColorMap(1);
			displayDepthOriginalCnnColorMap.displayColorMat();
		}



		if(useCnnNoWeigthUpdateJSONFile){
	     	cnn->copyInputMap2InputLayer(inputImageCnn);
		   cnn->forwardPassCnn();
			cnn->extractDepthMapCnn();
			cnn->setScaleDepthMap(scaleOriginalCnnMap);
			cnn->computeMeanDepthMap();
			cnn->replaceNegativeDepths();
			cnn->getCnnOutputMap().copyTo(depthMapToBeMerged);
			displayDepthOriginalCnnColorMap.setMap(cnn->getCnnOutputMap(), "Original CNN Depth Map");
			displayDepthOriginalCnnColorMap.setScaleFactor(255.0);
			displayDepthOriginalCnnColorMap.useColorMap(1);
			displayDepthOriginalCnnColorMap.displayColorMat();
			performanOriginalCnnMap = new manageDepthMapPerformance;
			performanOriginalCnnMap->setDepthMapGroundTruth(depthGT);
			performanOriginalCnnMap->setDepthMapEstimation(cnn->getCnnOutputMap());
			performanOriginalCnnMap->setScaleDepthMap(scaleOriginalCnnMap);
			performanOriginalCnnMap->setScaleGroundTruth(scaleInputDepthMap);
			performanOriginalCnnMap->computePerformance();
			costsFile << performanOriginalCnnMap->getLinearRMSE() << " ";
		}

		if(useCnnSslJSONFile){
			solver->copyInputMap2InputLayer(inputImageCnn);
			solver->copyGroundTruthInputMap2GroundTruthInputLayer(depthGT);
			solver->setScaleDepthMap(scaleSSLCnnMap);
			solver->forwardPassCnn();
			solver->extractDepthMapCnn();
			solver->computeMeanDepthMap();
			solver->replaceNegativeDepths();
			solver->getCnnOutputMap().copyTo(depthMapToBeMerged);
		    displayDepthCnnColorMap.setMap(solver->getCnnOutputMap(), "SSL CNN Depth Map");
			displayDepthCnnColorMap.setScaleFactor(255.0);
			displayDepthCnnColorMap.useColorMap(1);
			displayDepthCnnColorMap.displayColorMat();
			performanCnnMap = new manageDepthMapPerformance;
			performanCnnMap->setDepthMapGroundTruth(depthGT);
			performanCnnMap->setDepthMapEstimation(solver->getCnnOutputMap());
			performanCnnMap->setScaleDepthMap(scaleSSLCnnMap);
			performanCnnMap->setScaleGroundTruth(scaleInputDepthMap);
			performanCnnMap->computePerformance();
			costsFile << performanCnnMap->getLinearRMSE() << " ";
		}

		if(mergeJSONFile){

		    depthCnn.setDepthMap(depthMapToBeMerged);
		    depthCnn.setThresholdFilter(thresholdConfidence);

		    if(mergeFromConfidenceMap){
				depthCnn.filterPixels2BeMerged(depthStereoOpenCv);
				inputConfidenceMap->updateInputMap();
				//inputConfidenceMap->displayInputMapResized();
			}

			else
				depthCnn.filterPixels2BeMerged();

			depthCnn.mergeDepthMap(depthStereoOpenCv, "facil", 1.0, scaleSSLCnnMap);
			depthCnn.refreshPixels2BeMerged();
			displayDepthOriginalCnnColorMap.setMap(depthCnn.getMergedDepthMap(), "Merged Depth Map");
			displayDepthOriginalCnnColorMap.setScaleFactor(255.0/scaleSSLCnnMap);
			displayDepthOriginalCnnColorMap.useColorMap(1);
			displayDepthOriginalCnnColorMap.displayColorMat();
			performanceMergedMap = new manageDepthMapPerformance;
			performanceMergedMap->setDepthMapGroundTruth(inputDepthMap->getInputMapResized());
			performanceMergedMap->setDepthMapEstimation(depthCnn.getMergedDepthMap());
			performanceMergedMap->setScaleDepthMap(1.0);
			performanceMergedMap->setScaleGroundTruth(scaleInputDepthMap);
			performanceMergedMap->computePerformance();
			costsFile <<  performanceMergedMap->getLinearRMSE(); 
		}

		    costsFile << std::endl;

		cv::waitKey(5);
	}

	std::cout << "Leaving SSL..." << std::endl;
	costsFile.close();

	return 0;

}




