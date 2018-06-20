'''
In this script we retrain a given layer or layers
with other layers frozen.
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
img_width, img_height = 224, 224

if "-f" in options:
    conv_model = applications.InceptionV3(include_top=False, weights='imagenet')
    for case in ["train","test"]:
        database = []
        for class_name in class_names:
            print("Calculating features for {} samples of class {}".format(case,class_name))
            source_folder = "data/{}/{}".format(case,class_name)
            target_folder = "features/teacher/{}/{}".format(case,class_name)
            for filename in os.listdir(source_folder):
                new_filename = re.sub(r".jpg$",".npy",filename)
                img_path="{}/{}".format(source_folder,filename)
                img =keras.preprocessing.image.load_img(img_path,target_size=(img_width,img_height))
                img = keras.preprocessing.image.img_to_array(img)
                img = keras.applications.inception_v3.preprocess_input(np.array([img]))
                features = conv_model.predict(img)[0]
                feature_path ="{}/{}".format(target_folder,new_filename)
                np.save(feature_path,features)
                database.append({
                    "image_path":img_path,
                    "label":class_name_indices[class_name],
                    "label_name":class_name,
                    "teacher_feature_path":feature_path})
        pickle.dump(database,open("{}_data.p".format(case),"wb"))


def batch_generator(folder,batch_size):
    while True:
        features = np.empty((batch_size,5,5,2048))
        labels = np.empty((batch_size,))
        for i in range(batch_size):
            random_label = np.random.randint(len(class_names))
            class_name = class_names[random_label]
            filenames = os.listdir("{}/{}".format(folder,class_name))
            random_filename = np.random.choice(filenames)
            sample = np.load("{}/{}/{}".format(folder,class_name,random_filename))
            features[i] = sample
            labels[i] = random_label
        labels = keras.utils.to_categorical(labels)
        yield (features,labels)



if "-t" in options:
    feature_shape=(5,5,2048)
    train_generator = batch_generator("features/teacher/train",32)
    test_generator = batch_generator("features/teacher/test",32)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=(5,5,2048)))
    top_model.add(Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation=None,kernel_regularizer=keras.regularizers.l2(0.0001),name="logits"))
    top_model.add(Softmax())
    top_model.compile(optimizer=keras.optimizers.RMSprop(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    top_model.fit_generator(train_generator,steps_per_epoch=100,validation_data=test_generator,validation_steps=10,epochs=10)   
    top_model.save("models/teacher_top_model.h5")




def nonrandom_batch_generator(folder,batch_size):
    
    paths_by_label = [ ["{}/{}/{}".format(folder,name,x) for x in os.listdir("{}/{}".format(folder,name))] for name in class_names]
    print(paths_by_label)
    paths,labels = [],[]
    for i,group in enumerate(paths_by_label):
        paths+=group
        labels+=[i for p in group]
    nbr_samples = len(paths)

    q,r = divmod(nbr_samples,batch_size)
    nbr_batches = q +(1 if r>0 else 0)
    for i in range(nbr_batches):
        batch_samples = [np.load(p) for p in paths[i*batch_size:(i+1)*batch_size]]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        yield (np.array(batch_samples),np.array(batch_labels))
                
if "-l" in options: # l for logits
    top_model = keras.models.load_model("models/teacher_top_model.h5")
    logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)

    df = pickle.load(open("train_data.p","rb"))
    for sample in df:
        feature_path = sample["teacher_feature_path"]
        features = np.load(feature_path)
        features = np.array([features])
        logits = logits_model.predict(features)[0]
        sample["teacher_logits"]=logits
    pickle.dump(df,open("train_data.p","wb"))

