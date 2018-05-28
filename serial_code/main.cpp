#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "io.h"
#include "calculations.h"

//bool tDebug = false;
#ifdef tDebug
    #define tDebug 1
#else
    #define tDebug 0
#endif

int main(int argc, const char * argv[])
{
    if(4 != argc)
    {
        printUsage();
        exit(0);
    }
    
    auto startLoading = std::chrono::high_resolution_clock::now();
    
    std::string trainingDataPath(argv[1]);
    std::string testDataPath(argv[2]);    
    
    Data training = readTrainingData(trainingDataPath);
    Data test     = readTestData(testDataPath);

    // std::cout<< "Test image name: " << test.imagename.size() << std::endl;
    
    cv::Mat A = training.images;
    //print_matrix_details(A, "I: ");
    
    auto finishLoading = std::chrono::high_resolution_clock::now();
    
    auto startTraining = std::chrono::high_resolution_clock::now();
    
    cv::Mat meanImage(1, A.cols, CV_64F);
    cv::reduce(A, meanImage, 0, 1); // mean for each column
    //print_matrix_details(A, "avg: ");
    for (int i = 0; i < A.rows; ++i)
    {
        A.row(i) -= meanImage;
        for (int j = 0; j < A.cols; ++j)
        {
            if (A.at<double>(i,j) < 0)
            {
                A.at<double>(i,j) = 0;
            }
        }
    }

    //print_matrix_details(A, "mean image: ");
    // Reduced covariance mx (see wikipedia). The transposed one is switched because we use the rows as images.
    cv::Mat S = A*A.t();
    // print_matrix_details(S, "covar: ");
    cv::Mat eigenvalues;
    cv::Mat eigenvectors;
    
    cv::eigen(S, eigenvalues, eigenvectors);
    
    double componentNumUserInput = std::atof(argv[3]);
    unsigned int componentNum;
    if(componentNumUserInput < 1.0)
    {
        double varianceThreshold = componentNumUserInput;
        componentNum = getSufficientComponentNum(eigenvalues, varianceThreshold);
    }
    else
    {
        componentNum = round(componentNumUserInput);
    }
    
    cv::Mat V = restoreEigenvectors(A, eigenvectors, componentNum);
    
    cv::Mat W = getWeights(A, V);
    
    auto finishTraining = std::chrono::high_resolution_clock::now();
    
    auto startClassifying = std::chrono::high_resolution_clock::now();

    // classification
    classifyImages(test, training.labels, meanImage, V, W );
    
    auto finishClassifying = std::chrono::high_resolution_clock::now();
    
    // print if code is build to find avg computation time    
    if (tDebug)
    {
        std::cout << static_cast<std::chrono::duration<double>>(finishLoading     - startLoading    ).count() << "\t";
        std::cout << static_cast<std::chrono::duration<double>>(finishTraining    - startTraining   ).count() << "\t";
        std::cout << static_cast<std::chrono::duration<double>>(finishClassifying - startClassifying).count() << "\t";
        std::cout << static_cast<std::chrono::duration<double>>(finishClassifying - startLoading    ).count() << std::endl;
    }
    else
    {
        match_result(test);    
        std::cout << "------- Timing Output (measured in seconds) ---------------------" << std::endl;
        std::cout << "Time to load Image: \t" << static_cast<std::chrono::duration<double>>(finishLoading     - startLoading    ).count() << std::endl;
        std::cout << "Time for training: \t" << static_cast<std::chrono::duration<double>>(finishTraining    - startTraining   ).count() << std::endl;
        std::cout << "Time for Classification:" << static_cast<std::chrono::duration<double>>(finishClassifying - startClassifying).count() << std::endl;
        std::cout << "Total Time: \t\t" << static_cast<std::chrono::duration<double>>(finishClassifying - startLoading    ).count() << std::endl;

    }
    
    
    return 0;
}

