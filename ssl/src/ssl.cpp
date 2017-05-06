#include "ssl.h"

int  main(int argc, char const *argv[])
{
	int activeWindow = 1;
	bool quit = false;
	manageObjectCnn cnn;
    manageObjectInputMap inputImage("image", cnn.getSizeInputLayer());
	manageObjectInputMap inputDepthMap("depth", cnn.getSizeOutputLayer());
	manageObjectInputMap inputConfidenceMap("confidence", cnn.getSizeOutputLayer());
	manageObjectDepthMap depthCnn;
	manageDepthMapPerformance * performanDepthMap;
	displayObjectDepthMap displayDepthColorMap;

	for(;;){

		inputImage.readInputMap();
		inputImage.resizeInputMap();

		inputDepthMap.readInputMap();
		inputDepthMap.resizeInputMap();

		inputConfidenceMap.readInputMap();
		inputConfidenceMap.resizeInputMap();

		if(!inputImage.testInputMapExists() || !inputDepthMap.testInputMapExists() ){
			std::cout << "Map not found. Leaving..." << std::endl;
			return(0);
		}

		cnn.copyInputMap2InputLayer(inputImage.getInputMapResized());
		cnn.forwardPassCnn();
		cnn.extractDepthMapCnn();

	    depthCnn.setDepthMap(cnn.getCnnOutputMap());
		depthCnn.filterPixels2BeMerged(inputConfidenceMap.getInputMapResized());
		depthCnn.mergeDepthMap(inputDepthMap.getInputMapResized(), "facil");

		performanDepthMap = new manageDepthMapPerformance;

		performanDepthMap->setDepthMapGroundTruth(inputDepthMap.getInputMapResized());
		performanDepthMap->setDepthMapEstimation(depthCnn.getMergedDepthMap());
		performanDepthMap->computePerformance();

		inputImage.updatePath2InputMap();
		inputDepthMap.updatePath2InputMap();
		inputConfidenceMap.updatePath2InputMap();
		depthCnn.refreshPixels2BeMerged();

		char pressedKey;
		pressedKey = cv::waitKey(40);	

		setActiveWindow(pressedKey, &activeWindow);

		switch(activeWindow){

			case 1:
				inputImage.displayInputMapResized();
				break;

			case 2:
				displayDepthColorMap.setMap(depthCnn.getMergedDepthMap(), "Merged Depth Map");
				displayDepthColorMap.setScaleFactor(255/10);
				displayDepthColorMap.useColorMap(1);
				displayDepthColorMap.displayColorMat();
				break;

			case 3:
				displayDepthColorMap.setMap(inputConfidenceMap.getInputMapResized(), "Confidence Stereo Depth Map");
				displayDepthColorMap.setScaleFactor(1);
				displayDepthColorMap.useColorMap(1);
				displayDepthColorMap.displayColorMat();
				break;

			case 4:
				displayDepthColorMap.setMap(inputDepthMap.getInputMapResized(), "Input Stereo Depth Map");
				displayDepthColorMap.setScaleFactor(1);
				displayDepthColorMap.useColorMap(1);
				displayDepthColorMap.displayColorMat();
				break;

			case -99:
				quit = true;
				break;

			//case 6:
				
				//break;

			default:
				displayDepthColorMap.displayColorMat();
				break;
		}
	
		delete performanDepthMap;

		if(quit)
			break;

	}

	std::cout << "Leaving..." << std::endl;

	return 0;

}




