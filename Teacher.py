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
#nbr_train_samples = sum(len(os.listdir("data/train/{}".format(x))) for x in labels)
#nbr_test_samples = sum(len(os.listdir("data/test/{}".format(x))) for x in labels)

if "-f" in options:
    conv_model = applications.InceptionV3(include_top=False, weights='imagenet')
    database = []
    for case in ["train","test"]:
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
                    "image":img_path,
                    "label":class_name_indices[class_name],
                    "label_name":class_name,
                    "features":feature_path})
    pickle.dump(database,open("data.p","wb"))


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
    #print(paths[:100])
    nbr_samples = len(paths)

    q,r = divmod(nbr_samples,batch_size)
    nbr_batches = q +(1 if r>0 else 0)
    for i in range(nbr_batches):
        batch_samples = [np.load(p) for p in paths[i*batch_size:(i+1)*batch_size]]
        batch_labels = labels[i*batch_size:(i+1)*batch_size]
        yield (np.array(batch_samples),np.array(batch_labels))
                
if "-l" in options: # l for logits
    #folder = "features/teacher/train"
    #paths_by_label = [ ["{}/{}/{}".format(folder,name,x) for x in os.listdir("{}/{}".format(folder,name))] for name in class_names]
    #paths,labels = [],[]
    #for i,group in enumerate(paths_by_label):
    #    paths+=group
    #    labels+=[i for p in group]

    feature_shape=(5,5,2048)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=(5,5,2048)))
    top_model.add(Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation=None,kernel_regularizer=keras.regularizers.l2(0.0001),name="logits"))
    top_model.add(Softmax())
    top_model.compile(optimizer=keras.optimizers.RMSprop(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)

    df = pickle.load(open("data.p","rb"))
    for sample in df:
        feature_path = sample["feature_path"]
        features = np.load(feature_path)
        features = np.array([features])
        logits = logits_model.predict(features)[0]
        sample["logits"]=logits
    pickle.dump(df,open("data.p","wb"))
   
    #name in class_names:
    #    source_folder=
    #test_generator = nonrandom_batch_generator("features/teacher/test",32)
    #logit_preds = [logits_model.predict_on_batch(b[0]) for b in test_generator]
    #logit_preds = np.concatenate(logit_preds)
    #np.save("teacher_logits.npy",logit_preds)
   


#
#
#top_model_weights_path = 'TransferWeights.h5'
#train_data_dir = 'data/train'
#test_data_dir = 'data/test'
#print("There are {} validation samples.".format(nb_validation_samples))
#epochs = 10
#batch_size = 16
#
#modelInput = Input(shape=(img_width,img_height,3))
#conv_model = applications.InceptionV3(include_top=False, weights='imagenet')
#x = conv_model(modelInput)

#modelOutput = top_model(x)
#full_model = keras.models.Model(inputs=modelInput,outputs=modelOutput)
##for layer in full_model.layers:
##    print(layer)
#
#print(conv_model.summary())
#print(top_model.summary())
#
#
#def fine_tune():
#    for L in conv_model.layers[:-1]:
#        L.trainable = False
#    for L in top_model.layers:
#        L.trainable = False
#    full_model.compile(optimizer=keras.optimizers.SGD(0),
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
#
#    train_gen = datagen.flow_from_directory("data/train",target_size=(150,150),batch_size = 16)
#    test_gen = datagen.flow_from_directory("data/test",target_size=(150,150),batch_size = 16)
#    full_model.fit_generator(train_gen,validation_data = test_gen)
#
#def save_transfer_features(r=0): # r is the number of extra layers to remove 
#
#def train_top_model():
#    train_data = np.load('train_features.npy')
#    train_labels = np.load("train_labels.npy") 
#    test_data = np.load('test_features.npy')
#    test_labels = np.load("test_labels.npy")
#    
#    top_model.compile(optimizer=keras.optimizers.RMSprop(),
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
#    top_model.fit(train_data, train_labels,
#                epochs=epochs,
#                batch_size=batch_size,
#                validation_data=(test_data, test_labels),
#                verbose=1)
#    top_model.save_weights(top_model_weights_path)
#
#if "-f" in options:
#    save_transfer_features()
#if "-t" in options:
#    train_top_model()
#if "-e" in options:
#    top_model.load_weights(top_model_weights_path)
#    full_model.compile(optimizer=keras.optimizers.SGD(0.1),
#                    loss='categorical_crossentropy',
#                    metrics=['accuracy'])
#    gen1 = datagen.flow_from_directory("data/test",target_size=(150,150),batch_size = 40)
#    images,labels = next(gen1)
#    print("The labels are {}".format(labels))
#    preds = full_model.predict(images)
#    print("The predictions are {}".format(preds))
#    generator = get_generator(test_data_dir,"categorical")
#    how_good = full_model.evaluate_generator(generator) 
#    print(how_good)
#
#if "-ft" in options:
#    fine_tune()
#
#if "-pl" in options: # predict logits
#    top_model_logits = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)
#    logitModelInput = Input(shape=(img_width,img_height,3))
#    x2 = conv_model(logitModelInput)
#    logitModelOutput = top_model_logits(x2)
#    logitModel = keras.models.Model(inputs=logitModelInput,outputs=logitModelOutput)
#    logitModel.predict
#
#    generator = get_generator("data/train",None)
#    teacher_logits = [logitModel.predict(next(generator)) for i in range(nb_train_samples // batch_size)]
#    teacher_logits = np.vstack(teacher_logits)
#    np.save("teacher_logits.npy",teacher_logits)
#
#



