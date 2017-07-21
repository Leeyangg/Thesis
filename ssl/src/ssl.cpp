#include "ssl.h"

void setupStart(){

	scaleGT = scaleGTJSONFile/255.0;
	scaleSSLCnnMap= scaleDepthMapSslJSONFile;
	scaleOriginalCnnMap = scaleDepthMapCnnNoUpdateJSONFile;

 if(useZedJSONFile){
		if(!zedSourceSdkJSONFile)
			mergeFromConfidenceMap = false;
		
		else
			mergeFromConfidenceMap = true;

		zedCamObject = new manageZEDObject;
		inputImage = new manageObjectInputMap("zed", resolutionInputMapsJSONFile, zedCamObject);
		inputDepthMap = new manageObjectInputMap("zed_depth", resolutionOutputMapsJSONFile,zedCamObject);
		inputConfidenceMap = new manageObjectInputMap("zed_confidence", resolutionOutputMapsJSONFile,zedCamObject);
		thresholdConfidence = 70;

	}

	performanceStereoMap = new manageDepthMapPerformance;

	if(useCnnSslJSONFile){
		solver = new manageObjectCnn("solver");
		performanCnnMap = new manageDepthMapPerformance;


	}

	if(mergeJSONFile){
			performanceMergedMap = new manageDepthMapPerformance;
			performanceSecondMap = new manageDepthMapPerformance;		
		}


	if(useCnnNoWeigthUpdateJSONFile){
		cnn = new manageObjectCnn("cnn");
		performanOriginalCnnMap = new manageDepthMapPerformance;	
	}
		
	inputConfidenceMap = new manageObjectInputMap("confidence", resolutionOutputMapsJSONFile);

	if(useImportFromFolderJSONFile){

		mergeFromConfidenceMap = false;

		if(useStereoPairJSONFile)
			inputImage = new manageObjectInputMap("stereo_pair", resolutionInputMapsJSONFile);
		else
			inputImage = new manageObjectInputMap("image", resolutionInputMapsJSONFile);

		inputDepthMap = new manageObjectInputMap("depth", resolutionOutputMapsJSONFile);
		cheapDepth = new manageObjectInputMap("cheap_depth", resolutionOutputMapsJSONFile);

		if(!stereoOpenCVJSONFile){
			mergeFromConfidenceMap = true;
		}

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
extern  bool facilJSONFile;
extern  bool mergeJSONFile;
extern bool stereoOpenCVJSONFile;
extern bool displayOutputsJSONFile;
extern int numberFirstFrameJSONFile;
extern int numberLastFrameJSONFile;
extern int thresholdConfidenceJSONFile;
extern float scaleGTJSONFile;
extern bool janivanecky;

int  main(int argc, char const *argv[])
{
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	std::ofstream costsFile;
	costsFile.open("costs.txt");
	cv::Mat inputImageCnn;
	cv::Mat depthGT(resolutionInputMapsJSONFile.height,resolutionInputMapsJSONFile.width, CV_32FC1,1.0), 
			cheapDepthMap(resolutionInputMapsJSONFile.height,resolutionInputMapsJSONFile.width, CV_32FC1,1.0), 
			sparseDepthMap(resolutionInputMapsJSONFile.height,resolutionInputMapsJSONFile.width, CV_32FC1,1.0), notSizeddepthGT;
	cv::Mat depthMapToBeMerged;
	cv::Mat depthStereoOpenCv;
	cv::Mat leftImage;
	cv::Mat rightImage;
	cv::Mat pointsForSSL;
	cv::Mat confidenceResized;
	double min, max;
	cv::Mat grayInputImageCnn;
	cv::Mat dst, edgesInputImageCnn, edgg;
	config::loadVariablesFromJson();
	setupStart();
	int currentFrame = -1;
	int totalFrames = numberLastFrameJSONFile - numberFirstFrameJSONFile ;
	int lowerBoundary = 0;
	bool quit = false;
	clock_t startTime;
	cv::Mat displayGT(resolutionOutputMapsJSONFile.height,resolutionOutputMapsJSONFile.width, CV_32FC1,1.0), 
			displayCheap(resolutionOutputMapsJSONFile.height,resolutionOutputMapsJSONFile.width, CV_32FC1,1.0);

	while(!quit){
		startTime = clock();
		currentFrame++;
		std::cout << "Frame " << currentFrame << "/" << totalFrames << std::endl;

		if(currentFrame >= totalFrames){
			quit = true;
			
		}

		if(useZedJSONFile){
			zedCamObject->grabFrame();
			zedCamObject->getLeftImage(false).copyTo(leftImage);
			zedCamObject->getRightImage(false).copyTo(rightImage);

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

		}

		if(useImportFromFolderJSONFile){

			inputImage->readInputMap();
			inputImage->resizeInputMap();
			inputImage->getInputMapResized().copyTo(inputImageCnn);
			inputImage->updateInputMap();
			inputImage->displayInputMapResized();
		
			cv::cvtColor( inputImageCnn, grayInputImageCnn, CV_BGR2GRAY);
			cv::blur( grayInputImageCnn, edgesInputImageCnn, cv::Size(3,3));
			cv::resize(grayInputImageCnn,grayInputImageCnn,resolutionOutputMapsJSONFile);
			cv::Canny(  grayInputImageCnn, edgesInputImageCnn, 25,70,3 );
 
			if(displayOutputsJSONFile)
			    cv::imshow( "Edges input image",edgesInputImageCnn);

			if(mergeFromConfidenceMap){
				inputConfidenceMap->readInputMap();
				inputConfidenceMap->resizeInputMap();

				if(displayOutputsJSONFile){
					inputConfidenceMap->displayInputMapResized();
				}

				inputConfidenceMap->updateInputMap();
				cv::resize(inputConfidenceMap->getInputMapResized(), confidenceResized, resolutionInputMapsJSONFile);
			}

			inputDepthMap->readInputMap();
			inputDepthMap->resizeInputMap();
			inputDepthMap->getInputMapResized().copyTo(depthGT);	
			inputDepthMap->updateInputMap();

			if(!stereoOpenCVJSONFile){
				cheapDepth->readInputMap();
				cheapDepth->resizeInputMap();
				cheapDepth->getInputMapResized().copyTo(cheapDepthMap);	
				cheapDepth->updateInputMap();
				
				performanceStereoMap->setDepthMapGroundTruth(depthGT);
				performanceStereoMap->setDepthMapEstimation(cheapDepthMap);
				performanceStereoMap->setScaleDepthMap(STEREO_PERFORMANCE );
				performanceStereoMap->setScaleGroundTruth(scaleGT);

				if(thresholdConfidenceJSONFile != 0)
					performanceStereoMap->computePerformance(inputConfidenceMap->getInputMapResized());

				else
					performanceStereoMap->computePerformance();



				costsFile << "stereo " << performanceStereoMap->getThresholdError() << " " << performanceStereoMap->getAbsoluteRelativeError()  << " " << performanceStereoMap->getSquaredRelativeError() 
				<< " " << performanceStereoMap->getLinearRMSE()  << " " << performanceStereoMap->getLogRMSE()  << " " << performanceStereoMap->getScaleInvariantError()  << std::endl;
	
			}

			if(displayOutputsJSONFile){
				for (int row = 0; row < depthGT.rows; row++)
				{ 
					for (int col = 0; col < depthGT.cols; col++)
					{
						displayGT.at<float>(row, col) = (-1*((float)depthGT.at<uchar>(row, col)) + 255.0);
					 	displayCheap.at<float>(row, col) = (-1*((float)cheapDepthMap.at<uchar>(row, col)) + 255.0);
					}				
				}

				displayDepthColorMap.setMapResolution(resolutionOutputMapsJSONFile);
				displayDepthColorMap.setMap(displayGT, "Ground Truth Stereo");	
				displayDepthColorMap.setScaleFactor(1.0);
				displayDepthColorMap.useColorMap(1);
				displayDepthColorMap.displayColorMat();
				displayCheapDepthColorMap.setMapResolution(resolutionOutputMapsJSONFile);
				displayCheapDepthColorMap.setMap(displayCheap, "Cheap Stereo");	
				displayCheapDepthColorMap.setScaleFactor(1.0);
				displayCheapDepthColorMap.useColorMap(1);
				displayCheapDepthColorMap.displayColorMat();

			}

			if(useStereoPairJSONFile){
				inputImage->getInputMapResized().copyTo(leftImage);
				inputImage->getRightMapResized().copyTo(rightImage);
			}

		}

		if(displayOutputsJSONFile)
			cv::imshow("Original left image", inputImageCnn);

		if(leftImage.at<float>(0,0) !=0){
			if(stereoOpenCVJSONFile){
				bmAlgorithm.setResolution(resolutionOutputMapsJSONFile);
				bmAlgorithm.setScaleDepthMap(700.262*0.120);
				bmAlgorithm.setLeftImage(leftImage);
				bmAlgorithm.setRightImage(rightImage);
				bmAlgorithm.computeDisparityMap();
				bmAlgorithm.computeAbsoluteDepthMap();
				bmAlgorithm.getAbsoluteDepthMapResized().copyTo(depthStereoOpenCv);

				if(displayOutputsJSONFile){
					displayDepthStereoMap.setMapResolution(resolutionOutputMapsJSONFile);
					displayDepthStereoMap.setMap(bmAlgorithm.getAbsoluteDepthMap(), "Depth OpenCVBM");
					displayDepthStereoMap.setScaleFactor(255.0/20.0);
					displayDepthStereoMap.useColorMap(1);
					displayDepthStereoMap.displayColorMat();
				}

				performanceStereoMap->setDepthMapGroundTruth(depthGT);
				performanceStereoMap->setDepthMapEstimation(depthStereoOpenCv);
				performanceStereoMap->setScaleDepthMap(1.0);
				performanceStereoMap->setScaleGroundTruth(scaleGT);
				performanceStereoMap->computePerformance();
				costsFile << "stereo " << performanceStereoMap->getThresholdError() << " " << performanceStereoMap->getAbsoluteRelativeError()  << " " << performanceStereoMap->getSquaredRelativeError() 
				<< " " << performanceStereoMap->getLinearRMSE()  << " " << performanceStereoMap->getLogRMSE()  << " " << performanceStereoMap->getScaleInvariantError()  << std::endl;
			
			}

			
			if(mergeFromConfidenceMap){
				if(!zedSourceSdkJSONFile){
					depthCnn.filterPixels2BeMerged(inputConfidenceMap->getInputMapResized(), thresholdConfidenceJSONFile);
///					depthCnn2.filterPixels2BeMerged(inputConfidenceMap->getInputMapResized(), thresholdConfidenceJSONFile);
				}

				else{

					cv::resize(zedCamObject->getConfidenceMap(), confidenceResized, resolutionInputMapsJSONFile);
					depthCnn.filterPixels2BeMerged(confidenceResized, thresholdConfidenceJSONFile);

				}

			}
			
			else
				depthCnn.filterPixels2BeMerged(depthStereoOpenCv);	

			depthCnn.getPointsForSSL().copyTo(pointsForSSL);

			if(useCnnNoWeigthUpdateJSONFile){
			  	cnn->copyInputMap2InputLayer(inputImageCnn);
				cnn->forwardPassCnn(NOT_LEARN);
				cnn->extractDepthMapCnn();
				cnn->setScaleDepthMap(scaleOriginalCnnMap);
				cnn->computeMeanDepthMap();
				cnn->replaceNegativeDepths();

				if(!useCnnSslJSONFile)
					cnn->getCnnOutputMap().copyTo(depthMapToBeMerged);

				if(displayOutputsJSONFile){
					displayDepthOriginalCnnColorMap.setMapResolution(resolutionOutputMapsJSONFile);	
					if(!janivanecky){
						cv::Mat displayCNN(resolutionOutputMapsJSONFile.height,resolutionOutputMapsJSONFile.width, CV_32FC1,1.0);
						for (int row = 0; row < depthMapToBeMerged.rows; row++)
						{ 
							for (int col = 0; col < depthMapToBeMerged.cols; col++)
							{
								displayCNN.at<float>(row, col) = (-1*((float)depthMapToBeMerged.at<float>(row, col)) + 1.0);
							}				
						}
					
						displayDepthOriginalCnnColorMap.setMap(displayCNN, "Original CNN Depth Map");
					}

					else
						displayDepthOriginalCnnColorMap.setMap(cnn->getCnnOutputMap(), "Original CNN Depth Map");
					
						displayDepthOriginalCnnColorMap.setScaleFactor(255.0);


					displayDepthOriginalCnnColorMap.useColorMap(1);
					displayDepthOriginalCnnColorMap.displayColorMat();
				}
	
				performanOriginalCnnMap->setDepthMapGroundTruth(depthGT);
				performanOriginalCnnMap->setDepthMapEstimation(cnn->getCnnOutputMap());
				performanOriginalCnnMap->setScaleDepthMap(scaleOriginalCnnMap);
				performanOriginalCnnMap->setScaleGroundTruth(scaleGT);
				performanOriginalCnnMap->computePerformance();
				costsFile  << "cnn "<<  performanOriginalCnnMap->getThresholdError() << " " << performanOriginalCnnMap->getAbsoluteRelativeError()  << " " << performanOriginalCnnMap->getSquaredRelativeError() 
				<< " " << performanOriginalCnnMap->getLinearRMSE()  << " " << performanOriginalCnnMap->getLogRMSE()  << " " << performanOriginalCnnMap->getScaleInvariantError()  << std::endl;
			}

	
	
			if(useCnnSslJSONFile){
				solver->copyInputMap2InputLayer(inputImageCnn);
				if(stereoOpenCVJSONFile)
					solver->copyGroundTruthInputMap2GroundTruthInputLayer(depthStereoOpenCv);

				else
					solver->copyGroundTruthInputMap2GroundTruthInputLayer(cheapDepthMap);

			 	solver->copySparseLayer(pointsForSSL);
				solver->setScaleDepthMap(scaleSSLCnnMap);

				if(stereoOpenCVJSONFile){
					if(bmAlgorithm.getAvailablePoints() > 50){
						solver->forwardPassCnn(LEARN);
					}
					else
						solver->forwardPassCnn(NOT_LEARN);
				}

				else{

					if(depthCnn.getNumberPixelsForMerge() > lowerBoundary){
						solver->forwardPassCnn(LEARN);
					}
					else
						solver->forwardPassCnn(NOT_LEARN);
				}
		
				solver->extractDepthMapCnn();
				solver->computeMeanDepthMap();
				solver->replaceNegativeDepths();
				solver->getCnnOutputMap().copyTo(depthMapToBeMerged);
				
				if(displayOutputsJSONFile){
					displayDepthCnnSSLColorMap.setMap(depthMapToBeMerged, "SSL CNN Depth Map");
					displayDepthCnnSSLColorMap.setScaleFactor(255.0);
					displayDepthCnnSSLColorMap.useColorMap(1);
					displayDepthCnnSSLColorMap.displayColorMat();
				}

				performanCnnMap->setDepthMapGroundTruth(depthGT);
				performanCnnMap->setDepthMapEstimation(depthMapToBeMerged);
				performanCnnMap->setScaleDepthMap(scaleSSLCnnMap);
				performanCnnMap->setScaleGroundTruth(scaleGT);
				performanCnnMap->computePerformance();
				costsFile << "ssl " <<  performanCnnMap->getThresholdError() << " " << performanCnnMap->getAbsoluteRelativeError()  << " " << performanCnnMap->getSquaredRelativeError() 
				<< " " << performanCnnMap->getLinearRMSE()  << " " << performanCnnMap->getLogRMSE()  << " " << performanCnnMap->getScaleInvariantError()  << std::endl;


			}

			if(mergeJSONFile){
				if(displayOutputsJSONFile && depthCnn.getNumberPixelsForMerge() >lowerBoundary)
					cv::imshow("Sparse Map", pointsForSSL);

				depthCnn.setDepthMap(depthMapToBeMerged);
				depthCnn.setThresholdFilter(thresholdConfidence);
			//	depthCnn2.setDepthMap(depthMapToBeMerged);
			//	depthCnn2.setThresholdFilter(thresholdConfidence);

				float scaleMergedMap;

				if(useCnnSslJSONFile)
					scaleMergedMap = scaleSSLCnnMap;
				else
					scaleMergedMap = scaleOriginalCnnMap;

				if(depthCnn.getNumberPixelsForMerge() > lowerBoundary){
						if(facilJSONFile){
							if(!mergeFromConfidenceMap)
								depthCnn.mergeDepthMap(depthStereoOpenCv, "facil", 1.0, scaleMergedMap);

							else
								depthCnn.mergeDepthMap(cheapDepthMap, "facil", 1.0, scaleMergedMap);	
						}

						else{
							//depthCnn2.mergeDepthMap(cheapDepthMap, "facil", 1.0, 20.0);	
							depthCnn.setEdgeMap(edgesInputImageCnn);
							depthCnn.setConfidenceMap(inputConfidenceMap->getInputMapResized());			
							depthCnn.mergeDepthMap(cheapDepthMap, "confMerger", 1.0, scaleMergedMap);	
						}

					if(displayOutputsJSONFile){
						displayMergedMap.setMapResolution(resolutionOutputMapsJSONFile);
						displayMergedMap.setMap(depthCnn.getMergedDepthMap(), "Merged Depth Map");
						displayMergedMap.setScaleFactor(1.0/scaleGT);
						displayMergedMap.useColorMap(1);
						displayMergedMap.displayColorMat();
						//displayMergedMap.saveMap(currentFrame);

						displaySecondMap.setMapResolution(resolutionOutputMapsJSONFile);
						displaySecondMap.setMap(depthCnn.getSecondMap(), "Second Merged Depth Map");
						displaySecondMap.setScaleFactor(1.0/scaleGT);
						displaySecondMap.useColorMap(1);
						displaySecondMap.displayColorMat();


					}

				}
				depthCnn.refreshPixels2BeMerged();
				performanceMergedMap->setDepthMapGroundTruth(depthGT);
				performanceMergedMap->setScaleGroundTruth(scaleGT);

				if(depthCnn.getNumberPixelsForMerge() > lowerBoundary){
					performanceMergedMap->setDepthMapEstimation(depthCnn.getMergedDepthMap());
					performanceMergedMap->setScaleDepthMap(1.0);
				}
				else{
					performanceMergedMap->setDepthMapEstimation(depthMapToBeMerged);

					if(useCnnSslJSONFile){
						performanceMergedMap->setScaleDepthMap(scaleSSLCnnMap);

					}
					else
						performanceMergedMap->setScaleDepthMap(scaleSSLCnnMap);
		
				}

				performanceMergedMap->computePerformance();
				costsFile  <<  "original " <<  performanceMergedMap->getThresholdError() << " " << performanceMergedMap->getAbsoluteRelativeError()  << " " << performanceMergedMap->getSquaredRelativeError() 
				<< " " << performanceMergedMap->getLinearRMSE()  << " " << performanceMergedMap->getLogRMSE()  << " " << performanceMergedMap->getScaleInvariantError()  << std::endl;

				performanceSecondMap->setDepthMapGroundTruth(depthGT);
				performanceSecondMap->setScaleGroundTruth(scaleGT);

				if(depthCnn.getNumberPixelsForMerge() > lowerBoundary){
					performanceSecondMap->setDepthMapEstimation(depthCnn.getSecondMap());
					performanceSecondMap->setScaleDepthMap(1.0);
				}
				else{
					performanceSecondMap->setDepthMapEstimation(depthMapToBeMerged);
					if(useCnnSslJSONFile)
						performanceSecondMap->setScaleDepthMap(scaleSSLCnnMap);
					else
						performanceSecondMap->setScaleDepthMap(scaleSSLCnnMap);
				}


				performanceSecondMap->computePerformance();
				costsFile <<  "second " <<  performanceSecondMap->getThresholdError() << " " << performanceSecondMap->getAbsoluteRelativeError()  << " " << performanceSecondMap->getSquaredRelativeError() 
				<< " " << performanceSecondMap->getLinearRMSE()  << " " << performanceSecondMap->getLogRMSE()  << " " << performanceSecondMap->getScaleInvariantError()  << std::endl;

			}

		}

		std::cout << "Minutes to go = "<<  ((double)totalFrames -currentFrame)*((clock() - startTime)/CLOCKS_PER_SEC)/60.0 << "\n";
		
		if(displayOutputsJSONFile)
			cv::waitKey(50);

	}

	if(useCnnSslJSONFile){
		std::cout << "Saving SSL snapshot...\n";
		solver->saveSnapshot();
	}

	std::cout << "Leaving SSL..." << std::endl;
	costsFile.close();

	return 0;

}




