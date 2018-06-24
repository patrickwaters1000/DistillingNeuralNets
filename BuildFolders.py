import sys, os

class_names = sys.argv[1:]

def new_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def new_dirs(L):
    for p in L:
        new_dir(p)

new_dirs(["models","test_code","features","data"])
#new_dirs(["features/teacher","features/student"])
for folder in ["data","features/teacher","features/student"]:
    for case in ["train","test"]:
        #p="{}/{}".format(folder,case)
        #new_dir(p)
        for name in class_names:
            p="{}/{}/{}".format(folder,case,name)
            new_dir(p)
