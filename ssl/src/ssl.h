//my includes
#include "manageObjectInputMap.h"
#include "manageObjectCnn.h"
#include "manageObjectDepthMap.h"
#include "manageDepthMapPerformance.h"
#include "stereoAlgorithms.h"
#include "loadConfiguration.h"


//C includes
#include <iostream>
#include <fstream>
#include <cmath> 

#define NYUDataset 0
#define MyDataset 1
#define ZED 2


void setupStart();

float scaleInputDepthMap;
float scaleOriginalCnnMap;
float scaleSSLCnnMap;
bool mergeFromConfidenceMap;
int  thresholdConfidence;
float stdNoise;
manageObjectCnn * solver;
manageObjectCnn * cnn;
manageObjectInputMap * inputImage;
manageObjectInputMap * inputImageRight;
manageObjectInputMap * inputDepthMap;
manageObjectInputMap * inputConfidenceMap;
manageObjectDepthMap depthCnn;
manageDepthMapPerformance * performanOriginalCnnMap;
manageDepthMapPerformance * performanCnnMap;
manageDepthMapPerformance * performanceMergedMap;
displayObjectDepthMap displayDepthColorMap;
displayObjectDepthMap displayDepthCnnColorMap;
displayObjectDepthMap displayDepthOriginalCnnColorMap;
stereoBMOpencv bmAlgorithm;
manageZEDObject * zedCamObject;

