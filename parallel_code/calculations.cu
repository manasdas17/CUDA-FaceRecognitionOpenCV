#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "data_struct.h"

// helper for CUDA error handling
#define CUDA_SAFE_CALL( call ) { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      
      if (abort)
      {
          exit(code);
      }
   }
}

// grid dim x * block dim x >= pixelNum
// creating the mean image (meanImage) of all (imageNum many) images (images) which have pixelNum number of pixels
__global__ void getMeanImage( const double* images, double* meanImage, std::size_t imageNum, std::size_t pixelNum );

// grid dim x * block dim x >= pixelNum
// subtracting an image (meanImage) from a (imageNum large) set of images (images), each having pixelNum many pixels
__global__ void subtractMean( double* images, const double* meanImage, std::size_t imageNum, std::size_t pixelNum );

// grid dim x >= imageNum
// grid dim y * block dim x >= pixelNum
// calculating the lower triangle of A*A^t.
// Since A*A^t is symmetric, the lower triangle perfectly specifies it. Needed for eigenvalue calculation.
// A is the set of images (subtracted the mean image from each) which contains imageNum images, each having pixelNum pixels.
// S is the lower triangle output.
__global__ void getLowerAAt( const double* A, double* S, std::size_t imageNum, std::size_t pixelNum );

// grid dim = 1, block dim = 1 (used to avoid copying back to host)
// calculates the necessary number of principal components based ont he user input
__global__ void getSufficientComponentNum(const double* eigenvalues, std::size_t* componentNum, std::size_t eigenRows, double epsilon);

// this reorders the eigenvalues in descending order too
// grid dim x >= componentNum
// grid dim y * block dim x >= pixelNum
// for speeding up eigen decomposition we used this trick: https://en.wikipedia.org/wiki/Eigenface#Computing_the_eigenvectors
// this function restores the original eigenvectors from the one we got after applying the trick
__global__ void restoreEigenvectors( const double* meanSubtractedImages
                                   , const double* reducedEigenvectors
                                   , double* restoredEigenvectors
                                   , std::size_t imageNum
                                   , std::size_t pixelNum
                                   , std::size_t componentNum
                                   );

// grid dim x * block dim x >= componentNum
// normalizing each eigenvector
__global__ void normalizeEigenvector(double* restoredEigenvectors, std::size_t pixelNum, std::size_t componentNum);

// grid dim x >= componentNum
// grid dim y * block dim x >= imageNum
// calculating the weights of each used principal component in each of the training images (minus the mean of them)
__global__ void getWeights( const double* restoredEigenvectors
                          , const double* meanSubtractedImages
                          , double* weights
                          , std::size_t imageNum
                          , std::size_t pixelNum
                          , std::size_t componentNum
                          );
                                
// grid dim x >= componentNum
// grid dim y * block dim x >= testImageNum
// calculating weights of the used principal components in each of the test images (minus the training mean)
__global__ void getTestWeights( const double* restoredEigenvectors
                              , const double* meanImage
                              , const double* testImages
                              , double* testWeights
                              , std::size_t testImageNum
                              , std::size_t pixelNum
                              , std::size_t componentNum
                              );

// grid dim x >= imageNum
// grid dim y * block dim x >= testImageNum
// calculating the distances between each test image weight vector and training image weight vector
__global__ void getDistances( const double* trainingWeights
                            , const double* testWeights
                            , double* distances
                            , std::size_t trainImageNum
                            , std::size_t testImageNum
                            , std::size_t componentNum
                            );

// grid dim x * block dim x >= testImageNum
// selecting the minimum distance from the test/training distances
// ultimately gives us the "closest" training image to each test image
__global__ void getMinDistanceIdx( const double* distances
                                 , unsigned int* minDistanceImages
                                 , std::size_t trainImageNum
                                 , std::size_t testImageNum
                                 );

// device function for calculating the l2 norm on a 'size' large array
__device__ double getL2Norm(const double* array, std::size_t size);

// checks whether any CUDA capable device is availale
void checkDeviceAvailability();

// gets max thread per block number on the first CUDA capable device
int getMaxtThreadsPerBlock();

// helper for cuBlas error handling
void checkCuSolverResult(cusolverStatus_t solverStatus);

/// images should be in row major order
/// componentNumIndicator is either 0              - all of them are used
///                             in (0,1) range - explanatory power based component selection
///                             or larger      - rounded to intiger and that many are used
extern "C"
void gpuAssistedClassification( Data* training
                              , Data* test
                              , double componentNumIndicator
                              )
{
    
    auto startTraining = std::chrono::high_resolution_clock::now();
    
    
    std::size_t imageNum     = training->size;
    std::size_t pixelNum     = training->width * training->height;
    std::size_t testImageNum = test->size;
    std::size_t testPixelNum = test->width * test->height;    
    
    checkDeviceAvailability();
    int maxThreadsPerBlock = getMaxtThreadsPerBlock();
    
    
    
    // preprocessing training images
    
    double** A = training->images;
    
    std::size_t imageBitSize       = pixelNum * sizeof(double);
    std::size_t trainingSetBitSize = imageNum * imageBitSize;
    
    //pushing training images to GPU
    double* gpuA = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuA, trainingSetBitSize) );
    for(std::size_t i = 0; i < imageNum; ++i)
    {
        CUDA_SAFE_CALL( cudaMemcpy(&(gpuA[i * pixelNum]), A[i], imageBitSize, cudaMemcpyHostToDevice) );
    }
    
    double* gpuMeanImage = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuMeanImage, imageBitSize) );
        
    int gridDim = std::ceil(static_cast<double>(pixelNum) / maxThreadsPerBlock);
    getMeanImage<<< gridDim, maxThreadsPerBlock>>>(gpuA, gpuMeanImage, imageNum, pixelNum);
        
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    
    subtractMean<<< gridDim, maxThreadsPerBlock >>>(gpuA, gpuMeanImage, imageNum, pixelNum);
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    
    
    
    // calculating eigenvectors
    
    std::size_t imageNumSquareSize = imageNum * imageNum * sizeof(double);
    
    //pushing training images to GPU
    double* gpuS = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuS, imageNumSquareSize) );
    
    int gridDimY = std::ceil(static_cast<double>(imageNum) / maxThreadsPerBlock);
    getLowerAAt<<< dim3(imageNum, gridDimY), maxThreadsPerBlock >>>(gpuA, gpuS, imageNum, pixelNum);
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );  
    
    // calculating eigenvectors
    cusolverDnHandle_t solver;
    cusolverStatus_t solverStatus;
    
    double *gpuEigenvalues;
    int *devInfo;
    double *gpuWorkspace;
    
    int workspaceSize = 0;
    solverStatus = cusolverDnCreate(&solver);
                                              
    checkCuSolverResult(solverStatus);
    
    CUDA_SAFE_CALL( cudaMalloc ((void**)&gpuEigenvalues, imageNum * sizeof(double)) );
    CUDA_SAFE_CALL( cudaMalloc ((void**)&devInfo ,sizeof(int)) );
    
    cusolverEigMode_t eigenvalueFlag = CUSOLVER_EIG_MODE_VECTOR; //eigenvalues are calculated too
    cublasFillMode_t fillMode = CUBLAS_FILL_MODE_LOWER; // only the lower triangle has to store information
    // compute buffer size and prepare workspace
    solverStatus = cusolverDnDsyevd_bufferSize( solver
                                              , eigenvalueFlag
                                              , fillMode
                                              , imageNum
                                              , gpuS
                                              , imageNum
                                              , gpuEigenvalues
                                              , &workspaceSize
                                              );
                                              
    checkCuSolverResult(solverStatus);
                                              
    CUDA_SAFE_CALL( cudaMalloc((void**)&gpuWorkspace, workspaceSize*sizeof(double)) );
    
    // eigenvectors are stored in the same container (gpuS is overwritten)
    // WARNING contrary to OpenCV, eigenvalues (and corresponding eigenvectors) are in ascending order
    solverStatus = cusolverDnDsyevd( solver
                                   , eigenvalueFlag
                                   , fillMode
                                   , imageNum
                                   , gpuS
                                   , imageNum
                                   , gpuEigenvalues
                                   , gpuWorkspace
                                   , workspaceSize
                                   , devInfo
                                   );
                      
    checkCuSolverResult(solverStatus);
    
    solverStatus = cusolverDnDestroy(solver);
                                
    checkCuSolverResult(solverStatus);
    
    double* gpuEigenvectors = gpuS; // only renaming
    
    
    
    // deciding on the number of used principal components
    
    std::size_t componentNum;
    if(componentNumIndicator < 1.0)
    {
        
        std::size_t* gpuComponentNum = nullptr;
        CUDA_SAFE_CALL( cudaMalloc((void **)&gpuComponentNum, sizeof(std::size_t)) );
        
        double varianceThreshold = componentNumIndicator;
        getSufficientComponentNum<<<1,1>>>(gpuEigenvalues, gpuComponentNum, imageNum, varianceThreshold);
    
        CUDA_SAFE_CALL( cudaPeekAtLastError() );
        
        CUDA_SAFE_CALL( cudaMemcpy(&componentNum, gpuComponentNum, sizeof(std::size_t), cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaFree(gpuComponentNum) );
    }
    else
    {
        componentNum = round(componentNumIndicator);
    }
    
    
    // restoring the eigenvectors to the needed form
    
    CUDA_SAFE_CALL( cudaFree(gpuEigenvalues) );
    
    
    double* gpuRestoredEigenvectors = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuRestoredEigenvectors, componentNum * (pixelNum) * sizeof(double)) );
    
    gridDimY = std::ceil(static_cast<double>(pixelNum) / maxThreadsPerBlock);
    restoreEigenvectors<<< dim3(componentNum, gridDimY), maxThreadsPerBlock >>>( gpuA
                                                                               , gpuEigenvectors
                                                                               , gpuRestoredEigenvectors
                                                                               , imageNum
                                                                               , pixelNum
                                                                               , componentNum
                                                                               );
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );

    CUDA_SAFE_CALL( cudaFree(gpuEigenvectors) );
                                                                               
    gridDim = gridDimY;
    normalizeEigenvector<<<gridDim, maxThreadsPerBlock>>>( gpuRestoredEigenvectors
                                                         , pixelNum
                                                         , componentNum
                                                         );
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    
    
    
    // calculating training weights
    
    double* gpuW = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuW, componentNum * imageNum * sizeof(double)) );
                                     
    gridDimY = std::ceil(static_cast<double>(imageNum) / maxThreadsPerBlock);
    getWeights<<<dim3(componentNum, gridDimY), maxThreadsPerBlock>>>( gpuRestoredEigenvectors
                                                                    , gpuA
                                                                    , gpuW
                                                                    , imageNum
                                                                    , pixelNum
                                                                    , componentNum
                                                                    );
    
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    
    CUDA_SAFE_CALL( cudaFree(gpuA) );
    
    
    auto finishTraining = std::chrono::high_resolution_clock::now();
    
    auto startClassifying = std::chrono::high_resolution_clock::now();
    
    // classification
    
    std::size_t testSetBitSize = testImageNum * imageBitSize; // test images are the same size as training images
    
    //pushing test images to GPU
    double* gpuTestImages = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuTestImages, testSetBitSize) );
    for(std::size_t i = 0; i < testImageNum; ++i)
    {
        CUDA_SAFE_CALL( cudaMemcpy(&(gpuTestImages[i * testPixelNum]), (test->images)[i], imageBitSize, cudaMemcpyHostToDevice) );
    }
    
    double* gpuTestWeights = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuTestWeights, componentNum * testImageNum * sizeof(double)) );
    
    gridDimY = std::ceil(static_cast<double>(testImageNum) / maxThreadsPerBlock);
    getTestWeights<<<dim3(componentNum, gridDimY), maxThreadsPerBlock>>>( gpuRestoredEigenvectors
                                                                        , gpuMeanImage
                                                                        , gpuTestImages
                                                                        , gpuTestWeights
                                                                        , testImageNum
                                                                        , testPixelNum
                                                                        , componentNum
                                                                        );
        
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
                                                                        
    CUDA_SAFE_CALL( cudaFree(gpuRestoredEigenvectors) );              
    CUDA_SAFE_CALL( cudaFree(gpuMeanImage) );
    
    
    
    
    // calculating closest training image
    
    double* gpuDistances = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuDistances, imageNum * testImageNum * sizeof(double)) );
    
    gridDimY = std::ceil(static_cast<double>(testImageNum) / maxThreadsPerBlock);
    getDistances<<<dim3(imageNum, gridDimY), maxThreadsPerBlock>>>(gpuW
                                                                  , gpuTestWeights
                                                                  , gpuDistances
                                                                  , imageNum
                                                                  , testImageNum
                                                                  , componentNum
                                                                  );
        
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
    
    unsigned int* gpuMinDistanceIdxs = nullptr;
    CUDA_SAFE_CALL( cudaMalloc((void **)&gpuMinDistanceIdxs, testImageNum * sizeof(unsigned int)) );
    
    gridDim = std::ceil(static_cast<double>(testImageNum) / maxThreadsPerBlock);
    getMinDistanceIdx<<<gridDim, maxThreadsPerBlock>>>( gpuDistances
                                                      , gpuMinDistanceIdxs
                                                      , imageNum
                                                      , testImageNum
                                                      );
        
    CUDA_SAFE_CALL( cudaPeekAtLastError() );
 
 
    //translating closest training image to a label
 
    cudaMemcpy(test->labels, gpuMinDistanceIdxs, testImageNum * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for(std::size_t i = 0; i < testImageNum; ++i)
    {
            test->labels[i] = training->labels[test->labels[i]];
    }
    
    CUDA_SAFE_CALL( cudaDeviceReset() );
    
    auto finishClassifying = std::chrono::high_resolution_clock::now();
    
    std::cout << static_cast<std::chrono::duration<double>>(finishTraining    - startTraining   ).count() << "\t";
    std::cout << static_cast<std::chrono::duration<double>>(finishClassifying - startClassifying).count() << "\t";
}

void checkDeviceAvailability()
{
    int deviceCount = 0;
    CUDA_SAFE_CALL( cudaGetDeviceCount(&deviceCount) );
    if(0 == deviceCount)
    {
        std::cout << "No CUDA capable devices were found." << std::endl;
        exit(2);
    }
}

int getMaxtThreadsPerBlock()
{
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0); // assuming there is only one device
    int maxThreadsPerBlock = properties.maxThreadsPerBlock;
    return maxThreadsPerBlock;
}

__global__ void getMeanImage( const double* images, double* meanImage, std::size_t imageNum, std::size_t pixelNum )
{
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col >= pixelNum)
    {
        return;
    }
    
    meanImage[col] = 0.0;
    for(std::size_t row = 0; row < imageNum; ++row)
    {
        meanImage[col] += images[row*pixelNum + col];
    }
    
    meanImage[col] /= imageNum;
}

__global__ void subtractMean( double* images, const double* meanImage, std::size_t imageNum, std::size_t pixelNum )
{
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col >= pixelNum)
    {
        return;
    }
    
    for(std::size_t row = 0; row < imageNum; ++row)
    {
        images[row*pixelNum + col] -= meanImage[col];
        
        if(images[row*pixelNum + col] < 0.0)
        {
            images[row*pixelNum + col] = 0.0;
        }
    }
}

__global__ void getLowerAAt( const double* A, double* S, std::size_t imageNum, std::size_t pixelNum )
{
    std::size_t row = blockIdx.x;
    std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(row >= imageNum || col >= imageNum)
    {
        return;
    }
    
    S[row * imageNum + col] = 0.0;
    for(std::size_t i = 0; i < pixelNum; ++i)
    {
        S[row * imageNum + col] += A[row * pixelNum + i] * A[col * pixelNum + i];
    }
}

__global__ void getSufficientComponentNum(const double* eigenvalues, std::size_t* componentNum, std::size_t eigenRows, double epsilon)
{
    double variance = 0;
    for(std::size_t i = 0; i < eigenRows; ++i)
    {
        variance += eigenvalues[i];
    }
    variance *= eigenRows;
    
    (*componentNum) = 1;
    double subVariance = eigenRows * eigenvalues[eigenRows - 1];
    double explanatoryScore = subVariance / variance;
    for(; (*componentNum) < eigenRows && explanatoryScore <= epsilon; (*componentNum) += 1)
    {
        subVariance += eigenRows * eigenvalues[eigenRows - (*componentNum) - 1];
        explanatoryScore = subVariance / variance;
    }
}

__global__ void restoreEigenvectors( const double* meanSubtractedImages
                                   , const double* reducedEigenvectors
                                   , double* restoredEigenvectors
                                   , std::size_t imageNum
                                   , std::size_t pixelNum
                                   , std::size_t componentNum
                                   )
{
    std::size_t row = blockIdx.x;
    std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(col >= pixelNum || row >= componentNum)
    {
        return;
    }
    
    restoredEigenvectors[row * pixelNum + col] = 0.0;
    for(std::size_t i = 0; i < imageNum; ++i)
    {
        restoredEigenvectors[row * pixelNum + col] += reducedEigenvectors[(imageNum - row - 1) * imageNum + i] * meanSubtractedImages[i * pixelNum + col];
    }
}

__global__ void normalizeEigenvector(double* restoredEigenvectors, std::size_t pixelNum, std::size_t componentNum)
{
    std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row >= componentNum)
    {
        return;
    }
    
    double norm = getL2Norm(&(restoredEigenvectors[row * pixelNum]), pixelNum);
    
    for (int i = 0; i < pixelNum; ++i)
    {
        restoredEigenvectors[row * pixelNum + i] /= norm;
    }
}

__global__ void getWeights( const double* restoredEigenvectors
                          , const double* meanSubtractedImages
                          , double* weights
                          , std::size_t imageNum
                          , std::size_t pixelNum
                          , std::size_t componentNum
                          )
{
    std::size_t row = blockIdx.x;
    std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(col >= imageNum || row >= componentNum)
    {
        return;
    }
    
    weights[row * imageNum + col] = 0.0;
    for(std::size_t i = 0; i < pixelNum; ++i)
    {
        weights[row * imageNum + col] += restoredEigenvectors[row * pixelNum + i] * meanSubtractedImages[col * pixelNum + i];
    }
}

__global__ void getTestWeights( const double* restoredEigenvectors
                              , const double* meanImage
                              , const double* testImages
                              , double* testWeights
                              , std::size_t testImageNum
                              , std::size_t pixelNum
                              , std::size_t componentNum
                              )
{
    std::size_t row = blockIdx.x;
    std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(col >= testImageNum || row >= componentNum)
    {
        return;
    }
    
    testWeights[row * testImageNum + col] = 0.0;
    for(std::size_t i = 0; i < pixelNum; ++i)
    {
        double testImagePixelValue = testImages[col * pixelNum + i] - meanImage[i];
        if(testImagePixelValue < 0.0)
        {
            testImagePixelValue = 0.0;
        }
        testWeights[row * testImageNum + col] += restoredEigenvectors[row * pixelNum + i] * (testImagePixelValue);
    }
}

__global__ void getDistances( const double* trainingWeights
                            , const double* testWeights
                            , double* distances
                            , std::size_t trainImageNum
                            , std::size_t testImageNum
                            , std::size_t componentNum
                            )
{
    std::size_t row = blockIdx.x;
    std::size_t col = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(col >= testImageNum || row >= trainImageNum)
    {
        return;
    }
    
    distances[row * testImageNum + col] = 0.0;
    for(std::size_t i = 0; i < componentNum; ++i)
    {
        distances[row * testImageNum + col] += fabs(trainingWeights[i * trainImageNum + row] - testWeights[i * testImageNum + col]);
    }
}

__global__ void getMinDistanceIdx( const double* distances
                                 , unsigned int* minDistanceImages
                                 , std::size_t trainImageNum
                                 , std::size_t testImageNum
                                 )
{
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(col >= testImageNum)
    {
        return;
    }
    
    double minDistance = DBL_MAX;
    unsigned int minDistanceImageIdx = 0;
    for(unsigned int i = 0; i < trainImageNum; ++i)
    {
        if(distances[i * testImageNum + col] < minDistance)
        {
            minDistance = distances[i * testImageNum + col];
            minDistanceImageIdx = i;
        }
    }
    
    minDistanceImages[col] = minDistanceImageIdx;
}


__device__ double getL2Norm(const double* array, std::size_t size)
{
    double norm = 0;
    for (int i = 0; i < size; ++i)
    {
        norm += std::pow(array[i], 2);
    }
    norm = std::sqrt(norm);
    
    return norm;
}

void checkCuSolverResult(cusolverStatus_t solverStatus)
{
    if(CUSOLVER_STATUS_SUCCESS != solverStatus)
    {
        std::cout << "Error during eigen calculation. Error code: " << solverStatus << std::endl;
        exit(3);
    }
}
