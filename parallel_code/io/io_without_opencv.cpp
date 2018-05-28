#include <iostream>
#include <fstream>
#include <string>
#include <utility>

#include "io.h"

void printUsage()
{
    std::cout << "Usage: <executable> <path to training data file> <path to recognition task file> <variance threshold (if value is less than 1) or number of principal components (otherwise)>" << std::endl;
}

// reads pgm to [0,1] normalized double array on the heap in row major order
double* readPgm(const std::string& imageFilePath)
{
    std::ifstream imageFile;
    imageFile.open(imageFilePath.c_str());
    
    std::string line;
    getline(imageFile, line); // header line
    getline(imageFile, line);
    std::size_t width = std::atoi(line.c_str());
    getline(imageFile, line);
    std::size_t height = std::atoi(line.c_str());
    getline(imageFile, line); // max value line

    double* image = new double[width*height];
    
    int intensity;
    std::size_t i = 0;
    while(imageFile >> intensity)
    {
        image[i++] = static_cast<double>(intensity) / 255.0;
    }
    
    imageFile.close();
    
    return image;
}

// returns width and height of image (in this order)
std::pair<std::size_t, std::size_t> getPgmDimensions(const std::string& imageFilePath)
{
    std::ifstream imageFile;
    imageFile.open(imageFilePath.c_str());
    
    std::string line;
    getline(imageFile, line); // header line
    getline(imageFile, line);
    std::size_t width = std::atoi(line.c_str());
    getline(imageFile, line);
    std::size_t height = std::atoi(line.c_str());
    
    imageFile.close();
    
    return std::pair<std::size_t, std::size_t>(width, height);
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
    std::size_t lineCount = countLines(inputFilePath);
    if(0 == lineCount)
    {
        exit(1);
    }
    Data dataset(lineCount);
    
    std::ifstream input;
    input.open(inputFilePath.c_str());
    
    for(std::size_t i = 0; i < lineCount; ++i)
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
        
        double* image = readPgm(imagePath);
        
        dataset.images[i] = image;
        dataset.labels[i] = label;
        
        if(0 == i)
        {
            std::pair<std::size_t, std::size_t> dimensions = getPgmDimensions(imagePath);
            std::size_t width = dimensions.first;
            std::size_t height = dimensions.second;
            dataset.width = width;
            dataset.height = height;
        }
    }
    
    input.close();
        
    return dataset;
}

/// images are stored as row vectors
Data readTestData(const std::string& inputFilePath)
{
    std::size_t lineCount = countLines(inputFilePath);
    if(0 == lineCount)
    {
        exit(1);
    }
    Data dataset(lineCount);
    
    std::ifstream input;
    input.open(inputFilePath.c_str());
    
    for(std::size_t i = 0; i < lineCount; ++i)
    {
        std::string imagePath;
        
        getline(input, imagePath);

        imagePath.erase(imagePath.find_last_not_of(" \n\r\t")+1);
        
        if(imagePath.length() == 0)
        {
            break;
        }
        
        double* image = readPgm(imagePath);

        dataset.images[i] = image;
        
        if(0 == i)
        {
            std::pair<std::size_t, std::size_t> dimensions = getPgmDimensions(imagePath);
            std::size_t width = dimensions.first;
            std::size_t height = dimensions.second;
            dataset.width = width;
            dataset.height = height;
        }
    }
    
    input.close();
    
    return std::move(dataset);
}

std::size_t countLines(const std::string& filePath)
{
    std::ifstream file;
    file.open(filePath.c_str());
    
    std::size_t  lines = 0;
    int current;
    int previous = '\n';

    current = file.get();
    while (!file.eof())
    {
        if ('\n' == current && '\n' != previous)
        {
            ++lines;
        }
            
        previous = current;
        current = file.get();
    }
    
    if ('\n' != previous)
    {
        ++lines;
    }

    file.close();
    
    return lines;
}
