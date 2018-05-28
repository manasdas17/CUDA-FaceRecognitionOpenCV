#ifndef CALCULATIONS_H
#define CALCULATIONS_H

#include "data_struct.h"

unsigned int getSufficientComponentNum( cv::Mat eigenvalues
                                      , double epsilon
                                      );

double getL2Norm( cv::Mat row );

cv::Mat restoreEigenvectors( cv::Mat meanSubtractedImages
                           , cv::Mat reducedEigenvectors
                           , unsigned int componentNum
                           );

cv::Mat getWeights( cv::Mat meanSubtractedImages
                  , cv::Mat eigenvectors
                  );

void preprocessTestImages( cv::Mat testImage
                         , const cv::Mat meanImage
                         );

unsigned int getLabel( cv::Mat testImage
                     , const std::vector<unsigned int>& labelMapping
                     , const cv::Mat meanImage
                     , const cv::Mat eigenvectors
                     , const cv::Mat W
                     );

std::vector<unsigned int>&  classifyImages( Data& testData
                                          , const std::vector<unsigned int>& labelMapping
                                          , const cv::Mat meanImage
                                          , const cv::Mat eigenvectors
                                          , const cv::Mat W
                                          );
void match_result(Data& test);
#endif //CALCULATIONS_H
