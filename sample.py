import cv2
import os

path = "FinalDataset/ZoomIn/1"

images = [i for i in os.listdir(path)]

image = cv2.imread(path + "/" + images[0])


print(image.shape)
cv2.imshow("image",image[:,:,2])
cv2.waitKey(0)