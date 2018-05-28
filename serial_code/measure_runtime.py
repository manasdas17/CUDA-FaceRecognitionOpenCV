#!/usr/bin/python
import os
import sys


# parameters:
#   - number of measurements to average
#   - path to the directory
#   - name of the executable
#   - used training input for the executable
#   - used testing input for the executable
#   - used 3rd parameter for executable (number of principal components or confidence percentage)
if __name__ == '__main__':
    measurements = [0,0,0,0]
    for i in range(int(sys.argv[1])):
        os.chdir(sys.argv[2])
        os.system(' '.join(sys.argv[3:]) + ' > tmp_measurements')
        output = open('tmp_measurements', 'r').read()
        result = [float(x) for x in output.replace('\n', '').split('\t')]
        #print(str(i + 1) + '\t' + str(result))
        print str(i + 1) + '\t' + str(result)
        measurements = [measurements[j] + result[j] for j in range(len(measurements))]
    os.remove('tmp_measurements')
    measurements = [x / int(sys.argv[1]) for x in measurements]
    #print('AVG loading images: ' + str(measurements[0]) + ' s | AVG training model: ' + str(measurements[1]) + ' s | AVG classifying images: ' + str(measurements[2]) + ' s | AVG total ' + str(measurements[3]) + ' s | using ' + sys.argv[1] + ' individual measurements')
    print 'AVG loading images: ' + str(measurements[0]) + ' s | AVG training model: ' + str(measurements[1]) + ' s | AVG classifying images: ' + str(measurements[2]) + ' s | AVG total ' + str(measurements[3]) + ' s | using ' + sys.argv[1] + ' individual measurements'
