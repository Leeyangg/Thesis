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
extern bool displayOutputsJSONFile;

int  main(int argc, char const *argv[])
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	std::ofstream costsFile;
	costsFile.open("costs.txt");
	cv::Mat inputImageCnn;
	cv::Mat depthGT, notSizeddepthGT;
	cv::Mat depthMapToBeMerged;
	cv::Mat depthStereoOpenCv;
	cv::Mat leftImage;
	cv::Mat rightImage;
	cv::Mat pointsForSSL;
	config::loadVariablesFromJson();
	setupStart();

	for(;;){

		if(useZedJSONFile){
			//zedCamObject->grabFrame();
			zedCamObject->getLeftImage(true).copyTo(leftImage);
			zedCamObject->getRightImage(true).copyTo(rightImage);


			if(zedSourceSdkJSONFile){

				zedCamObject->getDepthMap().copyTo(depthGT);	

				if(displayOutputsJSONFile){	
						displayDepthCnnColorMap.setMap(depthGT, "ZED Depth Map");
						displayDepthColorMap.setScaleFactor(1.0/255.0);
						displayDepthCnnColorMap.useColorMap(1);
						displayDepthCnnColorMap.displayColorMat();
				}
			
			   cv::resize(depthGT,depthGT,resolutionOutputMapsJSONFile);

			}
	
			cv::resize(zedCamObject->getImage(), inputImageCnn, resolutionInputMapsJSONFile);
			if(displayOutputsJSONFile)
				cv::imshow("Original left image", inputImageCnn);

		}

		if(useImportFromFolderJSONFile){

			inputImage->readInputMap();
			inputImage->resizeInputMap();
			inputImage->getInputMapResized().copyTo(inputImageCnn);

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
			if(displayOutputsJSONFile){
				displayDepthColorMap.setMap(inputDepthMap->getInputMap(), "Input Stereo Depth Map");	
				displayDepthColorMap.setScaleFactor(1.0);
				displayDepthColorMap.useColorMap(1);
				displayDepthColorMap.displayColorMat();
			}

			if(useStereoPairJSONFile){
				inputImage->getInputMapResized().copyTo(leftImage);
				inputImage->getRightMapResized().copyTo(rightImage);
			}

		}

		if(leftImage.at<float>(0,0) !=0){
			if(stereoOpenCVJSONFile){
				bmAlgorithm.setResolution(resolutionOutputMapsJSONFile);
				bmAlgorithm.setScaleDepthMap(700.262*0.120);
				bmAlgorithm.setLeftImage(leftImage);
				bmAlgorithm.setRightImage(rightImage);
				bmAlgorithm.computeDisparityMap();
				bmAlgorithm.computeAbsoluteDepthMap();
				bmAlgorithm.getAbsoluteDepthMapResized().copyTo(depthStereoOpenCv);

				pointsForSSL = bmAlgorithm.getPointsForSSL();
				cv::resize(pointsForSSL,pointsForSSL,resolutionOutputMapsJSONFile);
				if(displayOutputsJSONFile){
					displayDepthStereoMap.setMapResolution(resolutionOutputMapsJSONFile);
					displayDepthStereoMap.setMap(bmAlgorithm.getAbsoluteDepthMap(), "Depth OpenCVBM");
					displayDepthStereoMap.setScaleFactor(255.0/10.0);
					displayDepthStereoMap.useColorMap(1);
					displayDepthStereoMap.displayColorMat();
				}

				performanceStereoMap = new manageDepthMapPerformance;
				performanceStereoMap->setDepthMapGroundTruth(depthGT);
				performanceStereoMap->setDepthMapEstimation(depthStereoOpenCv);
				performanceStereoMap->setScaleDepthMap(1.0);
				performanceStereoMap->setScaleGroundTruth(scaleInputDepthMap);
				performanceStereoMap->computePerformance();
				costsFile << "stereo " << performanceStereoMap->getLinearRMSE() << " ";
				free(performanceStereoMap);				
			}



			if(useCnnNoWeigthUpdateJSONFile){
			  	cnn->copyInputMap2InputLayer(inputImageCnn);
				cnn->forwardPassCnn();
				cnn->extractDepthMapCnn();
				cnn->setScaleDepthMap(scaleOriginalCnnMap);
				cnn->computeMeanDepthMap();
				cnn->replaceNegativeDepths();
				cnn->getCnnOutputMap().copyTo(depthMapToBeMerged);
				if(displayOutputsJSONFile){
					displayDepthOriginalCnnColorMap.setMap(cnn->getCnnOutputMap(), "Original CNN Depth Map");
					displayDepthOriginalCnnColorMap.setScaleFactor(255.0);
					displayDepthOriginalCnnColorMap.useColorMap(1);
					displayDepthOriginalCnnColorMap.displayColorMat();
				}
				performanOriginalCnnMap = new manageDepthMapPerformance;
				performanOriginalCnnMap->setDepthMapGroundTruth(depthGT);
				performanOriginalCnnMap->setDepthMapEstimation(cnn->getCnnOutputMap());
				performanOriginalCnnMap->setScaleDepthMap(scaleOriginalCnnMap);
				performanOriginalCnnMap->setScaleGroundTruth(scaleInputDepthMap);
				performanOriginalCnnMap->computePerformance();
				costsFile << "cnn "<< performanOriginalCnnMap->getLinearRMSE() << " ";
				free(performanOriginalCnnMap);
			}

			if(useCnnSslJSONFile){
				solver->copyInputMap2InputLayer(inputImageCnn);

//double min, max;
//cv::minMaxLoc(depthStereoOpenCv, &min, &max); std::cout << min << " " << max << std::endl;

				solver->copyGroundTruthInputMap2GroundTruthInputLayer(depthStereoOpenCv);
			 	solver->copySparseLayer(pointsForSSL);
				solver->setScaleDepthMap(scaleSSLCnnMap);
				solver->forwardPassCnn();
				solver->extractDepthMapCnn();
				solver->computeMeanDepthMap();
				solver->replaceNegativeDepths();
				solver->getCnnOutputMap().copyTo(depthMapToBeMerged);
				if(displayOutputsJSONFile){
					displayDepthCnnSSLColorMap.setMap(solver->getCnnOutputMap(), "SSL CNN Depth Map");
					displayDepthCnnSSLColorMap.setScaleFactor(255.0);
					displayDepthCnnSSLColorMap.useColorMap(1);
					displayDepthCnnSSLColorMap.displayColorMat();
				}
				performanCnnMap = new manageDepthMapPerformance;
				performanCnnMap->setDepthMapGroundTruth(depthGT);
				performanCnnMap->setDepthMapEstimation(solver->getCnnOutputMap());
				performanCnnMap->setScaleDepthMap(scaleSSLCnnMap);
				performanCnnMap->setScaleGroundTruth(scaleInputDepthMap);
				performanCnnMap->computePerformance();
				costsFile << "SSL" << performanCnnMap->getLinearRMSE() << " ";
				free(performanCnnMap);
			}

			if(mergeJSONFile){

				 depthCnn.setDepthMap(depthMapToBeMerged);
				 depthCnn.setThresholdFilter(thresholdConfidence);

				 if(mergeFromConfidenceMap){
					depthCnn.filterPixels2BeMerged(depthStereoOpenCv);
				//	inputConfidenceMap->updateInputMap();
					//inputConfidenceMap->displayInputMapResized();
				}

				else
					depthCnn.filterPixels2BeMerged();

				depthCnn.mergeDepthMap(depthStereoOpenCv, "facil", 1.0, scaleSSLCnnMap);
				depthCnn.refreshPixels2BeMerged();
				if(displayOutputsJSONFile){
					displayMergedMap.setMap(depthCnn.getMergedDepthMap(), "Merged Depth Map");
					displayMergedMap.setScaleFactor(255.0/scaleSSLCnnMap);
					displayMergedMap.useColorMap(1);
					displayMergedMap.displayColorMat();
				}
				performanceMergedMap = new manageDepthMapPerformance;
				performanceMergedMap->setDepthMapGroundTruth(depthGT);
				performanceMergedMap->setDepthMapEstimation(depthCnn.getMergedDepthMap());
				performanceMergedMap->setScaleDepthMap(1.0);
				performanceMergedMap->setScaleGroundTruth(scaleInputDepthMap);
				performanceMergedMap->computePerformance();
				costsFile <<  "merger " << performanceMergedMap->getLinearRMSE() << "\n"; 
				free(performanceMergedMap);
			}
				 costsFile << std::endl;
		}
		
		cv::waitKey(75);

	}

	std::cout << "Leaving SSL..." << std::endl;
	costsFile.close();

	return 0;

}




