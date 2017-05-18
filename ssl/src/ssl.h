//my includes
#include "manageObjectInputMap.h"
#include "manageObjectCnn.h"
#include "manageObjectDepthMap.h"
#include "manageDepthMapPerformance.h"

//C includes
#include <iostream>
#include <fstream>
#include <thread>

#define NYUDataset 0
#define MyDataset 1
#define ZED 2

void grabFrameZed(manageZEDObject* zedCamObject);
void setupStart(int choice);

float scaleInputDepthMap;
float scaleOriginalCnnMap;
float scaleSSLCnnMap;
bool mergeFromConfidenceMap;
int  thresholdConfidence;

manageObjectCnn * solver; //("solver");
manageObjectCnn * cnn;//cnn("cnn");
manageObjectInputMap * inputImage;//("image", solver.getSizeInputLayer());
manageObjectInputMap * inputDepthMap;//("depth", solver.getSizeOutputLayer());
manageObjectInputMap * inputConfidenceMap;//("confidence", solver.getSizeOutputLayer());
manageObjectDepthMap depthCnn;
manageDepthMapPerformance * performanOriginalCnnMap;
manageDepthMapPerformance * performanCnnMap;
manageDepthMapPerformance * performanceMergedMap;
displayObjectDepthMap displayDepthColorMap;
displayObjectDepthMap displayDepthCnnColorMap;
displayObjectDepthMap displayDepthOriginalCnnColorMap;
displayObjectDepthMap displayNoisyDepthMap;
manageZEDObject * zedCamObject;