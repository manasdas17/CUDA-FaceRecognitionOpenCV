#include <iostream>
#include <string>
#include <chrono>

#include "io.h"

/// images should be in row major order
/// componentNumIndicator is either 0              - all of them are used
///                              in (0,1) range - explanatory power based component selection
///                              or larger      - rounded to intiger and that many are used
extern "C" void gpuAssistedClassification( Data* training
                                         , Data* test
                                         , double componentNumIndicator
                                         );

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
    
    auto finishLoading = std::chrono::high_resolution_clock::now();
    
    std::cout << static_cast<std::chrono::duration<double>>(finishLoading - startLoading).count() << "\t";
    
    double componentNumUserInput = std::atof(argv[3]);
    
    gpuAssistedClassification(&training, &test, componentNumUserInput);
    
    auto finishTotal = std::chrono::high_resolution_clock::now();
   
    for(std::size_t i = 0; i < test.size; ++i)
    {
       std::cout << i << ".\t-\t" << test.labels[i] << std::endl;
    }
    
    
    std::cout << static_cast<std::chrono::duration<double>>(finishTotal   - startLoading).count() << "\n";
    
    return 0;
}
