import os
import shutil 

def rename():
    minCount = 1000
    maxCount = 0
    names = [i for i in os.listdir("ResizedDataset")]
    for name in names:
        gestures = [i for i in os.listdir("ResizedDataset/"+name)]
        # print(gestures)
        for gesture in gestures:
            folders = [j for j in os.listdir("ResizedDataset/"+name+"/"+gesture)]
            # print(folders)
            for folder in folders:
                lis = [i for i in os.listdir("ResizedDataset/"+name+"/"+gesture+"/"+folder)]
                # print("ResizedDataset/"+name+"/"+gesture+"/"+folder, len(lis))
                if len(lis) <  15:
                    # os.rmdir("ResizedDataset/"+name+"/"+gesture+"/"+folder)
                    # print(print("ResizedDataset/"+name+"/"+gesture+"/"+folder + " deleted"))
                    print("ResizedDataset/"+name+"/"+gesture+"/"+folder, len(lis))
                    # continue
                if len(lis) > maxCount:
                    maxCount = len(lis)
                if len(lis) < minCount:
                    minCount = len(lis)
                # for filename in os.listdir("ResizedDataset/"+name+"/"+gesture+"/"+folder): 
                    # datetime_str = filename[:-4]
                    # datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
                    # millisec = datetime_object.timestamp() * 1000
                    # print(filename)
                    # os.rename("CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+filename,"CapturedImages/"+name+"/"+gesture+"/"+folder+"/"+str(millisec) + ".jpg")
    print(minCount,maxCount)

def divide_chunks(l, n): 

    for i in range(0, len(l), n):  
        yield l[i:i + n] 
  

def arrange():
    writePath = "FinalDataset/"
    names = [i for i in os.listdir("ResizedDataset")]
    # names = ["Akhitha"]
    for name in names:
        gestures = [i for i in os.listdir("ResizedDataset/"+name)]
        # gestures = ["Capture"]
        # print(gestures)
        for gesture in gestures:
            # print(gesture)
            dest_dir = writePath + gesture
            writeFolderVal = len([i for i in os.listdir("FinalDataset/" + gesture)])
            print(writeFolderVal)

            folders = [j for j in os.listdir("ResizedDataset/"+name+"/"+gesture)]
            # print(folders)
            for folder in folders:
                lis = [i for i in os.listdir("ResizedDataset/"+name+"/"+gesture+"/"+folder)]
                # print(lis)

                for i in range(len(lis)-15):
                    dest_dir = writePath + gesture +  "/" + str(writeFolderVal + 1)
                    os.mkdir(dest_dir)
                    mylis = lis[i:i+15]
                    for file in mylis:
                        shutil.copy2("ResizedDataset/"+name+"/"+gesture+"/"+folder+"/"+file, dest_dir)
                        print(dest_dir + "/" + str(file))
                    writeFolderVal +=1
                

def check():
    gestures = [i for i in os.listdir("FinalDataset")]
    for gesture in gestures:
        folders = [i for i in os.listdir("FinalDataset/" + gesture)]
        for folder in folders:
            if len([i for i in os.listdir("FinalDataset/"+gesture+"/"+folder)]) != 5:
                print(gesture,folder)

if __name__ == '__main__':
    check()