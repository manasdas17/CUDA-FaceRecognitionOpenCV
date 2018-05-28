=======================
Using makefile to compile and execute code:
1. make -> to compile the code
2. make run -> to execute the code. Uses training.txt as training dataset and test_full.txt as test dataset with 12 PCA
3. make avgtime -> to execute code 10 times and compute average time
4. make clean -> clean executable files

=======================
used dataset:
	For the simplification of the image loading we used the same images in .pgm format P2 subformat (instead of the original P5). This basically contains the whole image as header information + list of pixel values in ascii encoding, instead the original which does the same in binary. After loading the images the difference between the source format is indistinguishable. We included both the new (P2 folder) and old (P5 folder) datasets. The data folder contains the P2 ones as well. Using P2 .pgm files does not require any modifications to the original OpenCV code.
	In the input folder we have also put two training/test files, these were the basis for the measurements. Be advised that the order of the images in them are shuffled. The P2 images also have different order / naming as the P5 images. You can get the label for an image with the following calculation: ((x-1)//10)+1 where x is the id in the name of the image, and // is the integer (truncating) division.
	
measurements:
	We used to setups:
		280 training images, 120 images to classify, use enough principal components to have 99% confidence, avg of 100 runs on the same random data
		390 training images,  10 images to classify, use enough principal components to have 99% confidence, avg of 100 runs on the same random data
	
	The results were the following:
		280/120 GPU:
		
			avg load time: 5.6s
			avg training time: 2.9s
			avg classification time: 0.36s
			avg total time: 8.9s
			(accuracy: 0.917)
		
		280/120 CPU:
		
			avg load time: 0.3s
			avg training time: 2.7s
			avg classification time: 0.0s
			avg total time: 3.0s
			(accuracy: 0.917)
		
		390/10 GPU:
		
			avg load time: 6.5s
			avg training time: 4.4s
			avg classification time: 0.3s
			avg total time: 11.2s
			(accuracy: 1.0)
		
		390/10 CPU:
		
			avg load time: 0.3s
			avg training time: 5.2s
			avg classification time: 0.0s
			avg total time: 5.5s
			(accuracy: 1.0)
			
	The load time only means the time we need to read the images from the files to the memory of the computer (not the GPU).
	Since clearly this part was making the biggest difference, We tried using the OpenCV provided functions too for image loading. We only tested this on the second setup (because in the first setup it is clear that the CPU version takes the prime). The results were the following:
	
		390/10 OpenCV/GPU:
		
			avg load time: 0.3s
			avg training time: 4.2s
			avg classification time: 0.3s
			avg total time: 4.8s
			(accuracy: 1.0)
	
compilation:
	
	CUDA 9.0 version is used to compile GPU code.

	
		nvcc -c ./calculations.cu -o ./calculations.o -L/usr/local/cuda/include
	    nvcc -ccbin g++ -m64 -Iusr/local/include/opencv2 -o pca.out calculations.o io.cpp main.cpp -lopencv_core -lopencv_imgcodecs -lcuda -lcudart -lcusolver -lcusparse -std=c++11
	
File structure:
	It is the same scheme as in the pure OpenCV one, only all the gpu related parts are now in the calculations.cu .
	In addition to the datasets we have added the 'io' folder which contains both versions (with and without) of the image loading. Just write over the original one, rename it and use the appropriate compilation command. The default one is the OpenCV version.
