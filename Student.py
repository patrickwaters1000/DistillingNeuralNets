'''
In this script we build the student model.
First use the -f option to calculate the features from the pretrained modiblenet.
Then build the top model, which can be trained from:
the teacher's logits using the -s option (student),
or the true labels using the -d option (delinquent)
'''
import os
import keras
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Softmax
from keras import applications
import re
import pickle

options = sys.argv[1:]
class_names = ["owl","pigeon"]
class_name_indices = {"owl":0, "pigeon":1}

if "-f" in options:
    def get_features(img_path):
        img = keras.preprocessing.image.load_img(img_path,target_size=(128,128))
        img = keras.preprocessing.image.img_to_array(img)
        img = keras.applications.mobilenet.preprocess_input(np.array([img]))
        return student_convnet.predict(img)[0]

    student_convnet = keras.applications.MobileNet(input_shape=(128,128,3),alpha=0.25,include_top=False,weights="imagenet")
    for case in ["train","test"]:
        df = pickle.load(open("{}_data.p".format(case),"rb"))
        for sample in df:
            img_path = sample["image_path"]
            feature_path = re.sub(r".jpg$",".npy",img_path)
            feature_path = re.sub(r"^data","features/student",feature_path)
            features = get_features(img_path)
            np.save(feature_path,features)
            sample["student_feature_path"]=feature_path
        pickle.dump(df,open("{}_data.p".format(case),"wb"))

if "-s" in options:
    def train_generator(df,batch_size,shuffle=True):
        feature_paths = np.array([s["student_feature_path"] for s in df])
        teacher_logits = np.array([s["teacher_logits"] for s in df])
        nbr_samples = len(feature_paths)
        nbr_batches = nbr_samples // batch_size
        print("The number of samples is {}".format(nbr_samples))
        while True:
            perm = np.arange(nbr_samples)
            if shuffle:
                np.random.shuffle(perm)
            
            for i in range(nbr_batches):
                indices = perm[i*batch_size:(i+1)*batch_size]
                batch_feature_paths = feature_paths[indices]
                batch_logits = teacher_logits[indices]
                batch_features = np.array([np.load(p) for p in batch_feature_paths])
                yield (batch_features,batch_logits)

    df = pickle.load(open("train_data.p","rb"))
    g = train_generator(df,32)
    
    top_model = keras.models.Sequential()
    top_model.add(Flatten(input_shape=(4,4,256)))
    top_model.add(Dense(200,activation="relu"))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(2,activation=None,name="logits"))
    top_model.add(Softmax())
    logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)
    logits_model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer = keras.optimizers.Adam())
    logits_model.fit_generator(g,steps_per_epoch=100,epochs=100)
    top_model.save("models/student.h5")


if "-e" in options:
     
    student_convnet = keras.applications.MobileNet(input_shape=(128,128,3),alpha=0.25,include_top=False,weights="imagenet")
    top_model = keras.models.load_model("models/student.h5")
    inp = Input(shape=(128,128,3))
    x = student_convnet(inp)
    out = top_model(x)
    full_model = keras.models.Model(inputs=inp,outputs=out)
    full_model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy,metrics=["acc"])

    def preprocess(img_path):
        img = keras.preprocessing.image.load_img(img_path,target_size=(128,128))
        img = keras.preprocessing.image.img_to_array(img)
        img = keras.applications.mobilenet.preprocess_input(np.array([img]))
        return img[0]

    def test_generator(df,batch_size,shuffle=True):
        image_paths = np.array([s["image_path"] for s in df])
        labels = keras.utils.to_categorical(np.array([s["label"] for s in df]))
        nbr_samples = len(image_paths)
        nbr_batches = nbr_samples // batch_size

        while True:
            perm = np.arange(nbr_samples)
            if shuffle:
                np.random.shuffle(perm)
            
            for i in range(nbr_batches):
                indices = perm[i*batch_size:(i+1)*batch_size]
                batch_img_paths = image_paths[indices]
                batch_images = np.array([preprocess(p) for p in batch_img_paths])
                batch_labels = labels[indices]
                yield (batch_images,batch_labels)

    df = pickle.load(open("test_data.p","rb"))
    g = test_generator(df,32,shuffle=True)
    how_good=full_model.evaluate_generator(g,steps=20)
    print("Here are the results from evaluating the student model-- {}".format(how_good))

if "-d" in options:
    def get_generator(df,batch_size,shuffle=True):
        feature_paths = np.array([s["student_feature_path"] for s in df])
        labels = np.array([s["label"] for s in df])
        labels = keras.utils.to_categorical(labels)
        nbr_samples = len(feature_paths)
        nbr_batches = nbr_samples // batch_size
        while True:
            perm = np.arange(nbr_samples)
            if shuffle:
                np.random.shuffle(perm)
            
            for i in range(nbr_batches):
                indices = perm[i*batch_size:(i+1)*batch_size]
                batch_feature_paths = feature_paths[indices]
                batch_labels = labels[indices]
                batch_features = np.array([np.load(p) for p in batch_feature_paths])
                yield (batch_features,batch_labels)

    df_train = pickle.load(open("train_data.p","rb"))
    df_test = pickle.load(open("test_data.p","rb"))
    g_train = get_generator(df_train,32)
    g_test = get_generator(df_test,32)
    
    top_model = keras.models.Sequential()
    top_model.add(Flatten(input_shape=(4,4,256)))
    top_model.add(Dense(200,activation="relu"))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(2,activation=None,name="logits"))
    top_model.add(Softmax())
    
    top_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(),
        metrics=["acc"])
    top_model.fit_generator(g_train,steps_per_epoch=100,validation_data=g_test,validation_steps=10,epochs=10)


