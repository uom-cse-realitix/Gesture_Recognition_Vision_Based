import cv2
import os
import math
import numpy as np

# img = cv2.imread("docs/images/hand.jpg")
# y = 100
# h = 50
# x = 100
# w = 50
# crop_img = img[y:y+h, x:x+w]
# # crop_img = img[:, :]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)




 
# img = cv2.imread('docs/images/hand.jpg', cv2.IMREAD_UNCHANGED)
 
# print('Original Dimensions : ',img.shape)
 
# width = 350
# height = 450
# dim = (width, height)
 
# # resize image
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
# print('Resized Dimensions : ',resized.shape)
 
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def getMaxSize():
    max_width = 0
    max_height = 0

    names = [i for i in os.listdir("FinalDataset")]
    print(names)
    path = "ResizedDataset"
    os.mkdir(path)
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
                    print(image.shape)

                    if image.shape[0] > max_height:
                        max_height = image.shape[0]
                    if image.shape[1] > max_width:
                        max_width = image.shape[1]
    return (max_height,max_width)

    

def getFolderSize():
    names = [i for i in os.listdir("ResizedDataset")]
    print(names)
    path = "ResizedDataset"
    # os.mkdir(path)
    for name in names:
        path = "ResizedDataset/" + name
        gestures = [i for i in os.listdir("ResizedDataset/"+name)]
        # os.mkdir(path)
        for gesture in gestures:
            path = "ResizedDataset/" + name + "/" + gesture
            folders = [j for j in os.listdir("ResizedDataset/"+name+"/"+gesture)]
            # os.mkdir(path)
            for folder in folders:
                path = "ResizedDataset/" + name + "/" + gesture + "/" + str(folder)
                # os.mkdir(path)
                lis = [i for i in os.listdir("ResizedDataset/"+name+"/"+gesture+"/"+folder)]
                if len(lis) < 15:
                    print(path, len(lis))
                    if len(lis) == 0:
                        os.rmdir("ResizedDataset/"+name+"/"+gesture+"/"+folder)

    return 

def crop():
    size = (100,100)

    #names = [i for i in os.listdir("FinalDataset") if i not in os.listdir("ResizedDataset")]
    names = ["Akhitha3"]
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


def main():
    # getMaxSize()
    # getFolderSize()
    crop()
                    
    
                    


if __name__ == '__main__':
    main()
                    
