#include "loadConfiguration.h"

using json = nlohmann::json;

bool useZedJSONFile;
bool zedSourceOpenCvJSONFile;
bool zedSourceSdkJSONFile;

bool useImportFromFolderJSONFile;
bool useStereoPairJSONFile;
std::string pathLeftImageJSONFile;
std::string pathRightImageJSONFile;
std::string formatNameImagesJSONFile;
std::string sourceGt;
std::string sourceCheapDepth;
std::string confidencePathJSONFile;
int numberFirstFrameJSONFile;
int numberLastFrameJSONFile;
int thresholdConfidenceJSONFile;

bool useCnnNoWeigthUpdateJSONFile;
std::string pathCaffemodelCnnJSONFile;
std::string pathProtofileCnnJSONFile;

bool useCnnSslJSONFile;
std::string pathCaffemodelSslCnnJSONFile;
std::string pathProtofileSslCnnJSONFile;
std::string pathSolverfileSslCnnJSONFile;

cv::Size resolutionInputMapsJSONFile;
cv::Size resolutionOutputMapsJSONFile;

float scaleDepthMapSslJSONFile;
float scaleDepthMapCnnNoUpdateJSONFile;

bool facilJSONFile;
bool mergeJSONFile;
bool stereoOpenCVJSONFile;
bool displayOutputsJSONFile;
bool janivanecky;
float scaleGTJSONFile;

namespace config{

	void loadVariablesFromJson(){

		json jsonFile;
		
		std::ifstream inputFile("../src/config/config.json");
		
		inputFile >> jsonFile;

		useZedJSONFile = jsonFile["/zed/use"_json_pointer];
		zedSourceOpenCvJSONFile = jsonFile["/zed/source/opencv"_json_pointer];
		zedSourceSdkJSONFile = jsonFile["/zed/source/sdk"_json_pointer];

		useImportFromFolderJSONFile = jsonFile["/importFromFolder/use"_json_pointer];
		useStereoPairJSONFile = jsonFile["/importFromFolder/stereoPair"_json_pointer];
		pathLeftImageJSONFile = jsonFile["/importFromFolder/source/left"_json_pointer];
		pathRightImageJSONFile = jsonFile["/importFromFolder/source/right"_json_pointer];
		formatNameImagesJSONFile = jsonFile["/importFromFolder/source/format"_json_pointer];
		numberFirstFrameJSONFile = jsonFile["/importFromFolder/source/firstFrame"_json_pointer];
		numberLastFrameJSONFile = jsonFile["/importFromFolder/source/lastFrame"_json_pointer];
		sourceGt = jsonFile["/importFromFolder/source/depthGt"_json_pointer];
		sourceCheapDepth = jsonFile["/importFromFolder/source/cheapDepth"_json_pointer];
		confidencePathJSONFile = jsonFile["/importFromFolder/source/confidence"_json_pointer];
		thresholdConfidenceJSONFile = jsonFile["/importFromFolder/source/thresholdConfidence"_json_pointer];

		janivanecky = jsonFile["/janivanecky"_json_pointer];

		useCnnNoWeigthUpdateJSONFile = jsonFile["/importCnnNoWeightUpdate/use"_json_pointer];
		pathCaffemodelCnnJSONFile = jsonFile["/importCnnNoWeightUpdate/source/caffemodel"_json_pointer];
		pathProtofileCnnJSONFile = jsonFile["/importCnnNoWeightUpdate/source/protofile"_json_pointer];

		useCnnSslJSONFile = jsonFile["/importSslCnn/use"_json_pointer];;
		pathCaffemodelSslCnnJSONFile = jsonFile["/importSslCnn/source/caffemodel"_json_pointer];
		pathProtofileSslCnnJSONFile = jsonFile["/importSslCnn/source/protofile"_json_pointer];
		pathSolverfileSslCnnJSONFile = jsonFile["/importSslCnn/source/solverfile"_json_pointer];

		resolutionInputMapsJSONFile.width = jsonFile["/resolutionInputMaps/width"_json_pointer];
		resolutionInputMapsJSONFile.height = jsonFile["/resolutionInputMaps/height"_json_pointer];

		resolutionOutputMapsJSONFile.width = jsonFile["/resolutionOutputMaps/width"_json_pointer];
		resolutionOutputMapsJSONFile.height = jsonFile["/resolutionOutputMaps/height"_json_pointer];

		scaleDepthMapCnnNoUpdateJSONFile = jsonFile["/scales/scaleOriginalCnnMap"_json_pointer];
		scaleDepthMapSslJSONFile = jsonFile["/scales/scaleSslCnnMap"_json_pointer];
		scaleGTJSONFile = jsonFile["/scaleGT"_json_pointer];
		
		facilJSONFile = jsonFile["/facil"_json_pointer];
		mergeJSONFile = jsonFile["/merge"_json_pointer];
		stereoOpenCVJSONFile = jsonFile["/stereoOPenCV"_json_pointer];
		displayOutputsJSONFile = jsonFile["/displayMaps"_json_pointer];
	}

}
