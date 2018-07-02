import os, sys

class_names = os.listdir("data/train")
for case in ["train","test"]:
    total = 0
    for name in class_names:
        
        x = len( os.listdir("data/{}/{}".format(case,name)) )
        total += x
        print("{} {} has {} images".format(case,name,x))
    print("Total of {} {} images".format(total,case))
