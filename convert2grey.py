import cv2
import os
import math
import numpy as np

def crop():
    size = (100,100)

    #names = [i for i in os.listdir("FinalDataset") if i not in os.listdir("ResizedDataset")]
    names = ["AkhithaTesting"]
    print(size)
    path = "ResizedDataset"
    # os.mkdir(path)
    for name in names:
        path = "ResizedDataset/" + name
        gestures = [i for i in os.listdir("FinalDataset/"+name)]
        os.mkdir(path)
        for gesture in gestures:
            path = "ResizedDataset/" + name + "/" + gesture
            folders = [j for j in os.listdir("FinalDataset/"+name+"/"+gesture)]
            os.mkdir(path)
            for folder in folders:
                path = "ResizedDataset/" + name + "/" + gesture + "/" + str(folder)
                os.mkdir(path)
                for filename in os.listdir("FinalDataset/"+name+"/"+gesture+"/"+folder): 
                    path = "ResizedDataset/" + name + "/" + gesture + "/" + str(folder) + "/" + filename
                    pathRead = "FinalDataset/" + name + "/" + gesture + "/" + str(folder) + "/" + filename
                    image = cv2.imread(pathRead)

                    resized = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
                    grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                    cv2.imwrite(path, grayImage)
                    print(path + " image saved")
    return

def convert():
    size = (100,100)
    #names = [i for i in os.listdir("Testing")]
    names = ["Test3"]
    for name in names:
        gestures = [i for i in os.listdir("Testing/" + name)]
        for gesture in gestures:
            folders = [i for i in os.listdir("Testing/" + name + "/" + gesture)]
            for folder in folders:
                for file in os.listdir("Testing/"+name + "/"  + gesture + "/" + folder):
                    path = "Testing/" + name + "/" + gesture + "/" + str(folder) + "/" + file

                    image = cv2.imread(path)
                    resized = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
                    grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    os.remove(path)
                    cv2.imwrite(path, grayImage)
                    print(path + " image saved")



if __name__ == '__main__':
    convert()
