#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H

#include <opencv2/opencv.hpp>

struct Data
{
    cv::Mat_<double> images;							// image matrix
    std::vector<unsigned int> labels;					// Actual label
    std::vector<unsigned int> classificationLabels;		// store label of matching classification
    std::vector<std::string> imagename;					// iamge name
};

#endif //DATA_STRUCT_H
