//my includes
#include "manageObjectInputMap.h"
#include "manageObjectCnn.h"
#include "manageObjectDepthMap.h"
#include "manageDepthMapPerformance.h"

void setActiveWindow(char pressedKey, int * activeWindow){

	switch(pressedKey){

		case '1':
		    cv::destroyAllWindows();
			*activeWindow = 1;
			break;

		case '2':
			cv::destroyAllWindows();
			*activeWindow = 2;
			break;

		case '3':
			cv::destroyAllWindows();
			*activeWindow = 3;
			break;

		case '4':
			cv::destroyAllWindows();
			*activeWindow = 4;
			break;

		case 'q':
			cv::destroyAllWindows();
			*activeWindow = -99;
			break;

		case '6':
			//cv::destroyAllWindows();
			//*activeWindow = 6;
			break;

		default:
			break;
	}
}
