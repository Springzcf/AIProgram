#coding=utf-8
import csv
import numpy as np

def writeData(trainFileName,testFileName,expressList,dataCount = 100):
    iLen = len(expressList)
    data = np.asarray(np.ones((dataCount,iLen)))
    orgX = np.asarray(list(range(1,dataCount+1,1)),'f')
    np.random.shuffle(orgX)
    for index in range(iLen):
        dataTemp = []
        if expressList[index] == 'x':
            np.random.shuffle(orgX)
            data[:,index] = orgX
        else:
            if index == 0:
                for x in orgX:
                    dataTemp.append(eval(expressList[index]))
            elif index == 1:
                for x in data[:,0]:
                    dataTemp.append(eval(expressList[index]))
            elif index == 2:
                for xy in data:
                    x = xy[0]
                    y = xy[1]
                    dataTemp.append(eval(expressList[index]))
            elif index == 3:
                for xyz in data:
                    x = xyz[0]
                    y = xyz[1]
                    z = xyz[2]
                    dataTemp.append(eval(expressList[index]))
            data[:,index] = dataTemp
    fr = open(trainFileName,'w',newline='')
    fr_test = open(testFileName,'w',newline='')
    fr_write = csv.writer(fr)
    fr_test_write = csv.writer(fr_test)
    fr_write.writerows(data[0:int(dataCount * 0.9)])
    fr_test_write.writerows(data[int(dataCount * 0.9):])
    fr.close()
    fr_test.close()

