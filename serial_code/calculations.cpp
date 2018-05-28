#include <cmath>
#include <vector>
#include <limits>
#include <opencv2/opencv.hpp>
#include "calculations.h"

#ifdef DEBUG
    #define DEBUG_PRINT printf
#else
    #define DEBUG_PRINT
#endif

unsigned int getSufficientComponentNum(cv::Mat eigenvalues, double epsilon)
{
    unsigned int n = eigenvalues.rows;
    
    double variance = 0;
    for(unsigned int i = 0; i < n; ++i)
    {
        variance += eigenvalues.at<double>(i, 0);
    }
    variance *= n;
    
    unsigned int componentNum = 1;
    double subVariance = n * eigenvalues.at<double>(0,0);
    double explanatoryScore = subVariance / variance;
    for(; componentNum < n && explanatoryScore <= epsilon; ++componentNum)
    {
        subVariance += n * eigenvalues.at<double>(componentNum);
        explanatoryScore = subVariance / variance;
    }
    
    return componentNum;
}

double getL2Norm(cv::Mat row)
{
    double result = 0;
    for (int i = 0; i < row.cols; ++i)
    {
        result += std::pow(row.at<double>(0,i), 2);
    }
    result = std::sqrt(result);
    
    return result;
}

cv::Mat restoreEigenvectors(cv::Mat meanSubtractedImages, cv::Mat reducedEigenvectors, unsigned int componentNum)
{
    cv::Mat restored(componentNum, meanSubtractedImages.cols, CV_64F);
    for (unsigned int i = 0; i < componentNum; ++i)
    {
        cv::Mat row = reducedEigenvectors.row(i)*meanSubtractedImages;
    
        row.copyTo(restored.row(i));
        double norm = getL2Norm(restored.row(i));
        restored.row(i) /= norm;
    }
    
    return restored;
}

cv::Mat getWeights(cv::Mat meanSubtractedImages, cv::Mat eigenvectors)
{
    cv::Mat W(eigenvectors.rows, meanSubtractedImages.rows, CV_64F);
    for (int i = 0; i < eigenvectors.rows; ++i)
    {
        for (int j = 0; j < meanSubtractedImages.rows; ++j)
        {
            cv::Mat tmp = eigenvectors.row(i)*meanSubtractedImages.row(j).t();
            W.at<double>(i,j) = tmp.at<double>(0,0);
        }
    }
    
    return W;
}

void preprocessTestImage(cv::Mat testImage, const cv::Mat meanImage)
{
    testImage -= meanImage;
    
    for (int i = 0; i < testImage.cols; ++i)
    {
        if (testImage.at<double>(0,i) < 0)
        {
            testImage.at<double>(0,i) = 0;
        }
    }
}

unsigned int getLabel( cv::Mat testImage
                     , const std::vector<unsigned int>& labelMapping
                     , const cv::Mat meanImage
                     , const cv::Mat eigenvectors
                     , const cv::Mat W
                     )
{
    int k = eigenvectors.rows;

    preprocessTestImage(testImage, meanImage);
    
    cv::Mat testWeights = cv::Mat::zeros(k, 1, CV_64F);
    for (int i = 0; i < k; ++i)
    {
        cv::Mat temp = eigenvectors.row(i)*testImage.t();
        testWeights.at<double>(i,0) = temp.at<double>(0,0);
    }
    
    double minDistance = std::numeric_limits<double>::infinity();
    unsigned int label = -1;
    for (int trainingImage = 0; trainingImage < W.cols; ++trainingImage)
    {
        double distance = 0;
        for (int component = 0; component < k; ++component)
        {
            distance += fabs(W.at<double>(component, trainingImage) - testWeights.at<double>(component,0));
        }
        
        if (distance < minDistance)
        {
            minDistance = distance;
            label = trainingImage;
        }
    }
    return labelMapping[label];
}

std::vector<unsigned int>& classifyImages( Data& testData
                                         , const std::vector<unsigned int>& labelMapping
                                         , const cv::Mat meanImage
                                         , const cv::Mat eigenvectors
                                         , const cv::Mat W
                                         )
{
    for(int i = 0; i < testData.images.rows; ++i)
    {
        cv::Mat testImage = testData.images.row(i);
        // testData.labels.push_back(getLabel(testImage, labelMapping, meanImage, eigenvectors, W));
        testData.classificationLabels.push_back(getLabel(testImage, labelMapping, meanImage, eigenvectors, W));
    }
    
    //return testData.labels;
    return testData.classificationLabels;
}

void match_result(Data& test)
{
    // std::cout << "Image Name" << "\t-" << "Match Label" << "\t-" << "Actual Label" << "\t-" << "Match/Not-Match" << std::endl;
    DEBUG_PRINT("Image Name \t- Match Label \t- Actual Label \t- Match/Not-Match");
    unsigned int match=0, noMatch=0;

    
    
    for(unsigned int i = 0; i < test.imagename.size(); ++i)
    {
        ((test.labels[i] == test.classificationLabels[i]) ? match++: noMatch++); 
        // std::cout << test.imagename[i] << "\t-\t" << test.classificationLabels[i] << "\t-\t" << test.labels[i] << "\t-\t"<< ((test.labels[i] == test.classificationLabels[i]) ? "Match": "Not-Match") << std::endl;
        // char str[20] = {test.imagename[i].c_str()};
        DEBUG_PRINT("%s \t- %d \t- %d \t- %s\n", test.imagename[i].c_str(), test.classificationLabels[i], test.labels[i], ((test.labels[i] == test.classificationLabels[i]) ? "Match": "Not-Match"));
    }

    std::cout << "Match: " << match << "\t Non-Match: " << noMatch << "\t Accuracy: " << (float) (match*100)/(match+noMatch) << "%" << std::endl; 
// 
}
