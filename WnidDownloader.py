import re, sys, os, subprocess
import urllib.request

class_names = ["Basenji","Basset","Beagle","BedlingtonTerrier","BerneseMountainDog"]
wnids = ['n02110806', 'n02088238', 'n02088364', 'n02093647', 'n02107683']


def new_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)



for folder in ["data","features/teacher","features/student"]:
    for case in ["train","test"]:
        for name in class_names:
            path = "{}/{}/{}".format(folder,case,name)
            new_dir(path)

#nbr_train = 200
#nbr_test = 100


def get_urls(wnid,name):
    save_path = "URL_lists/{}_URLs".format(name)
    if not os.path.exists(save_path):
        new_dir("URL_lists")
        image_locations_url="http://image-net.org/api/text/imagenet.synset.geturls?wnid="+wnid
        urllib.request.urlretrieve(image_locations_url,save_path)
    
    urls = open(save_path,"r").readlines()
    urls = [re.sub("\n","",u) for u in urls]
    return urls

for name,wnid in zip(class_names,wnids):
    urls = get_urls(wnid,name)
    url_gen = (url for url in urls)
    for case in ["train","test"]:
        print("Case {}, nbr_train {}, nbr_test {}".format(case,nbr_train,nbr_test))
        max_images = {"train":nbr_train,"test":nbr_test}[case]
        print("Downloading at most {} {} images".format(max_images,name))
        for i in range(10000):
            try:
                url = next(url_gen)
                save_path="data/{}/{}/{}{:05d}.jpg".format(case,name,name,i)
                subprocess.call(["./Downloader.sh",url,save_path])
            except StopIteration:
                break
