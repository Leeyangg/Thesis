//my includes
#include "manageObjectInputMap.h"
#include "manageObjectCnn.h"
#include "manageObjectDepthMap.h"
#include "manageDepthMapPerformance.h"

//C includes
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>

#define NYUDataset 0
#define MyDataset 1

void grabFrameZed(manageZEDObject* zedCamObject);
void setupStart(int choice);

float scaleInputDepthMap;
float scaleOriginalCnnMap;
float scaleSSLCnnMap;
bool mergeFromConfidenceMap;