import os
from PIL import Image
for case in ["train","test"]:
    for name in os.listdir("data/{}".format(case)):
        fnames = os.listdir("data/{}/{}".format(case,name))
        paths = ["data/{}/{}/{}".format(case,name,fname) for fname in fnames]
        for p in paths:
            try:
                Image.open(p)
            except OSError:
                os.remove(p)
            
