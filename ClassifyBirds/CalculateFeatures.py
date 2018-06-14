from keras.applications import inception_v3, mobilenet
from keras.preprocessing import image
import numpy as np
from PIL import Image
import os, re

animals = open("animals","r").read().split("\n")

model_names = ["mobilenet","inception"]
models = {	"mobilenet": mobilenet.MobileNet(weights = 'imagenet', include_top=False),
		"inception": inception_v3.InceptionV3(weights = 'imagenet', include_top=False)		}
p_funcs = {	"mobilenet": mobilenet.preprocess_input, 
		"inception": inception_v3.preprocess_input		}

if not os.path.exists("Features"):
	os.makedirs("Features")
for animal in animals:
	if not os.path.exists("Features/"+animal):
		os.makedirs("Features/"+animal)
	img_folder= "Images/"+animal
	img_paths = [img_folder+"/"+x for x in os.listdir(img_folder)]
	images = []
	for p in img_paths:
		try:
			img = image.load_img(p)
			img = img.resize((224,224),Image.ANTIALIAS)
			img = image.img_to_array(img)
			images.append(img)
		except IOError:
			pass
	images = np.array(images)
	for model_name in model_names:
		images2 = p_funcs[model_name](images)
		features = models[model_name].predict(images2)
		features = features.reshape((len(images2),-1))
		np.save("Features/{}/{}.npy".format(animal,model_name),features)
		print("Finished calulating {} features for class {}".format(model_name,animal))

