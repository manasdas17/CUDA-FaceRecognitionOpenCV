#ifndef IO_H
#define IO_H

#include "data_struct.h"

void printUsage();

/// preprocessing image to the form factor needed by PCA
void pgmToMatRow(cv::Mat& image);
void print_matrix_details(cv::Mat& matrix, std::string matrix_name);

cv::Mat readPgm(const std::string& imagePath);

/*
    assumptions:
        - inputFile is having training examples in the following format <absolute path to image>\t<label (nonnegative number)>
        - each image is already resampled to the needed size
        - each image has the same dimensions
        - there is at least one training image
        
    images are stored as rows
*/
Data readTrainingData(const std::string& inputFilePath);

/// images are stored as row vectors
Data readTestData(const std::string& inputFilePath);

#endif //IO_H
