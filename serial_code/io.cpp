#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "io.h"

void printUsage()
{
    std::cout << "Usage: <executable> <path to training data file> <path to recognition task file> <variance threshold (if value is less than 1) or number of principal components (otherwise)>" << std::endl;
}

// utility function to print matrix information
void print_matrix_details(cv::Mat& matrix, std::string matrix_name)
{
    double minVal; 
    double maxVal;
    minMaxLoc( matrix, &minVal, &maxVal);
    
    std::cout << "Details of " << matrix_name << " => ";
    std::cout << "rows: " <<matrix.rows << " || cols: " << matrix.cols << " || min val: " << minVal << " || max val: " << maxVal << std::endl;
}

/// preprocessing image to the form factor needed by PCA
void pgmToMatRow(cv::Mat& image)
{
    const double intensity_range = 255.0; // range of the possible value in the used image format
    
    image.convertTo(image, CV_64F, 1.0 / intensity_range); // convert input image format to a standard one
    image = image.reshape(0, 1); // we have the images as row vectors
}

cv::Mat readPgm(const std::string& imagePath)
{
    cv::Mat image = imread(imagePath, cv::IMREAD_UNCHANGED);
    if(image.data)
    {
        pgmToMatRow(image);
    }
    
    return image;
}

/*
    assumptions:
        - inputFile is having training examples in the following format <absolute path to image>\t<label (nonnegative number)>
        - each image is already resampled to the needed size
        - each image has the same dimensions
        - there is at least one training image
        
    images are stored as rows
*/
Data readTrainingData(const std::string& inputFilePath)
{
    Data dataset;
    
    std::ifstream input;
    input.open(inputFilePath.c_str());
    
    while(!input.eof())
    {
        std::string imagePath;
        std::string labelString;
        unsigned int label;
        
        getline(input, imagePath, '\t');
        imagePath.erase(imagePath.find_last_not_of(" \n\r\t")+1);
        
        if(imagePath.length() == 0)
        {
            break;
        }
        
        getline(input, labelString);
        label = std::atoi(labelString.c_str());
        
        cv::Mat image = readPgm(imagePath);

        if(image.data)
        {
            dataset.images.push_back(image);
            dataset.labels.push_back(label);
        }
        else
        {
             std::cout << "WARNING: image not loaded! Image path is: " << imagePath << std::endl;
        }
    }
    
    input.close();
    
    return dataset;
}

/// images are stored as row vectors
Data readTestData(const std::string& inputFilePath)
{
    Data dataset;
    
    std::ifstream input;
    input.open(inputFilePath.c_str());
    
    while(!input.eof())
    {
        std::string imagePath;
        std::string labelString;
        unsigned int label;
        
        // getline(input, imagePath);
        getline(input, imagePath, '\t');

        imagePath.erase(imagePath.find_last_not_of(" \n\r\t")+1);
        
        if(imagePath.length() == 0)
        {
            break;
        }

        getline(input, labelString);
        label = std::atoi(labelString.c_str());
        
        
        cv::Mat image = readPgm(imagePath);

        if(image.data)
        {
            // std::cout << "imagepath: "<< imagePath << " - \t" << "label test data: " << label << std::endl;
            dataset.images.push_back(image);
            dataset.labels.push_back(label);
            dataset.imagename.push_back(imagePath);
        }
        else
        {
             std::cout << "WARNING: image not loaded! Image path is: " << imagePath << std::endl;
        }
    }
    
    dataset.labels.reserve(dataset.images.rows);
    
    input.close();
    
    return dataset;
}
