Usage:
    Overwrite the files in the original projects (see files to use in the folders).
    Call the provided python file. You can call it like the following:
    
        python measure_runtime.py 100 "/home\usernam\foldername" ./pca.out ./input/training.txt ./input/test.txt 0.99
        
    where the role of the arguments are:
        python - calling the python interpreter from command line
        measure_runtime.py - name of the python file
        100 - number of measurements you want to run consecutively (they will all be printed out and their average in the end too)
        "/home\usernam\foldername" - base path to the project folder
        pca.out - name of the executable
        ./input/training.txt - first argument of pca.out
        ./input/test.txt - second argument of pca.out
        0.99 - third argument of pca.out
        
note:
    I suggest that at first you run the projects one time (by hand or just passing 1 to the measuring program) and start measuring only after that. This first time can be a lot slower because binaries and the CUDA runtime has to load first. After this first time however the programs maintain a stable time of running.

     
