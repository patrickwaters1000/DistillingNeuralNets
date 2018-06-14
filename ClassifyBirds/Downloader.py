'''
Gets downloads a list of URLs for the names stored in "animals",
then calls a bash script to do the actual downloading
'''
import re, sys
import urllib.request
import os
import subprocess

animals = open("animals","r").readlines()
animals = [re.sub("\n","",x) for x in animals]
print(animals)
max_nbr = str(1000)
for word in animals:
	wnid_data = open("index.noun","r").read()
		
	str1 = r'^q .*$'
	str2 = re.sub('q',word,str1)
	hits = re.findall(str2,wnid_data,re.MULTILINE)
	print(hits)
	assert len(hits)==1
	ids = re.findall(r'[\d]{8,8}',hits[0])
	wnid = 'n'+ids[0]
	print(wnid)
	
	image_locations_url="http://image-net.org/api/text/imagenet.synset.geturls?wnid="+wnid
	urllib.request.urlretrieve(image_locations_url,"{}URLs".format(word))
	print("URLs retrieved")
	
	subprocess.call(["./Downloader.sh",word,max_nbr])
	os.remove("{}URLs".format(word))
