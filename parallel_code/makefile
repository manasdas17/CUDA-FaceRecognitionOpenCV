# stop echo of command on termianl
.SILENT: all run debug clean avgtime
.PHONY: clean

CXX = g++ -m64 -std=c++11

# cuda c++ compiler
CUDAC = nvcc -c -std=c++11 -Wno-deprecated-gpu-targets
CUDAC1 = nvcc -Wno-deprecated-gpu-targets -ccbin 
#Compilation flag
CXXFLAGS = `pkg-config --cflags opencv`
#Linking flag
LDFLAGS = -Iusr/local/include/opencv2 -lopencv_core -lopencv_imgcodecs -lcuda -lcudart -lcusolver -lcusparse
CUDA_LDFLAG = -L/usr/local/cuda/include

# Executable file
TARGET = main
CUDA_TARGET = calculations

#source files
SOURCE = $(CUDA_TARGET).o io.cpp main.cpp
CUDA_SOURCE = $(CUDA_TARGET).cu

#
all : 
	echo "Building code ... "
	$(CUDAC) -o $(CUDA_TARGET)".o" $(CUDA_SOURCE) $(CUDA_LDFLAG)
	$(CUDAC1) $(CXX) -o $(TARGET)".out" $(SOURCE) $(LDFLAGS)
	echo "Build complete!! Type make run to execute the code."

# to print debug messages
debug : $(SOURCE)
	echo "Building code in debug mode... "
	$(CUDAC) -o $(CUDA_TARGET)".o" $(CUDA_SOURCE) $(CUDA_LDFLAG)
	$(CUDAC1) $(CXX) -o $(TARGET)".out" $(SOURCE) $(LDFLAGS)
	echo "Build complete!! Type make run to execute the code."

# to print debug messages
avgtime : $(SOURCE)
	echo "Building code in debug mode... "
	$(CUDAC) -DtDebug -o $(CUDA_TARGET)".o" $(CUDA_SOURCE) $(CUDA_LDFLAG)
	$(CUDAC1) $(CXX) -o $(TARGET)".out" $(SOURCE) $(LDFLAGS)
	echo "Build complete!! Computing avg computation time."
	# running python script to run the program for 10 times and computing average time
	python measure_runtime.py 10 . ./main.out ./input/training.txt ./input/test_full.txt 12

clean:
	echo "Cleaning Up ... "
	rm -f *.o *.out *~ $(TARGET)
	echo "Done"

run:
	./$(TARGET)".out" ./input/training.txt ./input/test_full.txt 12