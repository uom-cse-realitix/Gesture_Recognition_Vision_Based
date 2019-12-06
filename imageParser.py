import os

names = [i for i in os.listdir("CapturedImages")]
for name in names:
    gestures = [i for i in os.listdir("CapturedImages/"+name)]
    print(gestures)
    for gesture in gestures:
        folders = [j for j in os.listdir("CapturedImages/"+name+"/"+gesture)]
        print(folders)
        for folder in folders:
            for filename in os.listdir("CapturedImages/"+name+"/"+gesture+"/"+folder): 
                print(filename)
                # continue