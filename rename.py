import os
from datetime import datetime
import time


# os.rename("CapturedImages/Chatta/Capture/0/2019-11-06 18:45:23.936696.jpg","CapturedImages/Chatta/Capture/0/2019-11-06 18:45:23.93.jpg")
# os.rename("Captured","CapturedImages")

                # continue

def rename():
    names = ["AkhithaTesting"]
    for name in names:
        gestures = [i for i in os.listdir("CapturedImages/"+name)]
        print(gestures)
        for gesture in gestures:
            folders = [j for j in os.listdir("CapturedImages/"+name+"/"+gesture)]
            print(folders)
            for folder in folders:
                for filename in os.listdir("CapturedImages/"+name+"/"+gesture+"/"+folder): 
                    datetime_str = filename[:-4]
                    datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                    millisec = datetime_object.timestamp() * 1000
                    print(datetime_str, millisec)
                    os.rename("CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+filename,"CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+str(millisec) + ".jpg")

def sortRename():
    minVal = 100
    maxVal = 0
    names = ["AkhithaTesting"]
    for name in names:
        gestures = [i for i in os.listdir("FinalDataset/"+name)]
        print(gestures)
        for gesture in gestures:
            folders = [j for j in os.listdir("FinalDataset/"+name+"/"+gesture)]
            print(folders)
            for folder in folders:
                for filename in os.listdir("FinalDataset/"+name+"/"+gesture+"/"+folder): 
                    datetime_str = filename[:-4]
                    # print(filename,datetime_str)
                    nms = datetime_str.split(".")
                    print(name,gesture,folder,filename,nms[-1])
                    if len(nms[-1]) < minVal:
                        minVal = len(nms[-1])
                    if len(nms[-1]) > maxVal:
                        maxVal = len(nms[-1])
                    # if len(nms[-1]) == 15:
                    #     print(name,gesture,folder,filename)
                        # os.rename("CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+filename,"CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+nms[-1][0:-7] +"." + nms[-1][-7:] + ".jpg")
                        # if nms[-1][-3:] == "001":
                        #     os.rename("CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+filename,"CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+nms[-1][0:] +"." + nms[1] + ".jpg")                   
                    if len(nms[-1]) < 6:
                        nms[-1] = nms[-1].ljust(6, '0') 
                        print("**************",nms[-1])
                    if len(nms[-1]) < 7:
                        nms[-1] = nms[-1].ljust(7,'1')
                        print(filename,nms[0]+"."+nms[1]) 
                        # os.rename("FinalDataset/"+name+"/"+gesture+"/"+folder+"/"+filename,"FinalDataset/"+name+"/"+gesture+"/"+folder+"/"+nms[0] +"." + nms[1] + ".jpg")
                    continue
    print(minVal,maxVal)
    return
                    




def main():
    sortRename()


if __name__ == '__main__':
    main()
