'''
Gets downloads a list of URLs for the names stored in "animals",
then calls a bash script to do the actual downloading
'''
import re, sys
import urllib.request
import os
import subprocess
import multiprocessing
import time

class_names = os.listdir("data/train")
print(class_names)
nbr_train = 10
nbr_test = 5

def get_wnid(name):
    wnid_data = open("index.noun","r").read()
    str1 = r'^q .*$'
    str2 = re.sub('q',name,str1)
    hits = re.findall(str2,wnid_data,re.MULTILINE)
    print(hits)
    assert len(hits)==1
    ids = re.findall(r'[\d]{8,8}',hits[0])
    wnid = 'n'+ids[0]
    print(wnid)
    return wnid

def get_urls(wnid):
    image_locations_url="http://image-net.org/api/text/imagenet.synset.geturls?wnid="+wnid
    urllib.request.urlretrieve(image_locations_url,"URLs")
    urls = open("URLs","r").readlines()
    urls = [re.sub("\n","",u) for u in urls]
    os.remove("URLs")
    return urls

def download(url,save_path,max_time=1.0):
    def d():
        #urllib.request.urlretrieve(url,save_path)
        f=open(save_path,"wb")
        f.write(requests.get(url).content)
        f.close()
    p=multiprocessing.Process(target=d)
    p.start()
    p.join(max_time)
    if p.is_alive():
        p.terminate()
        p.join()

for case in ["train","test"]:
    for name in class_names:
        wnid = get_wnid(name)
        urls = get_urls(wnid)
        for i,url in enumerate(urls):
            save_path="data/{}/{}/{}{:06d}.jpg".format(case,name,name,i)
            subprocess.call(["./Downloader.sh",url,save_path])
            if i>=nbr_train:
                break
