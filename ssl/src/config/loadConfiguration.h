//My includes
#include "json.h"

// OpenCV includes
#include "opencv2/opencv.hpp"

//C includes
#include <fstream>  

extern  bool useZedJSONFile;
extern  bool zedSourceOpenCvJSONFile;
extern  bool zedSourceSdkJSONFile;

extern  bool useImportFromFolderJSONFile;
extern  bool useStereoPairJSONFile;
extern  std::string pathLeftImageJSONFile;
extern  std::string pathRightImageJSONFile;
extern  std::string formatNameImagesJSONFile;
extern  std::string sourceGt;
extern  std::string confidencePathJSONFile;
extern  int numberFirstFrameJSONFile;

extern  bool useCnnNoWeigthUpdateJSONFile;
extern  std::string pathCaffemodelCnnJSONFile;
extern  std::string pathProtofileCnnJSONFile;

extern  bool useCnnSslJSONFile;
extern  std::string pathCaffemodelSslCnnJSONFile;
extern  std::string pathProtofileSslCnnJSONFile;
extern  std::string pathSolverfileSslCnnJSONFile;

extern  cv::Size resolutionInputMapsJSONFile;
extern  cv::Size resolutionOutputMapsJSONFile;

extern float scaleDepthMapSslJSONFile;
extern float scaleDepthMapCnnNoUpdateJSONFile;

extern bool mergeJSONFile;
extern bool stereoOpenCVJSONFile;
extern bool displayOutputsJSONFile;

namespace config{

	void loadVariablesFromJson();

}
