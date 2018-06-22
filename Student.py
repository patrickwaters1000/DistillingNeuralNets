'''
In this script we build the student model.
First use the -f option to calculate the features from the pretrained modiblenet.
Then build the top model, which can be trained from:
the teacher's logits using the -s option (student),
or the true labels using the -d option (delinquent)
'''
import os, sys, re, pickle, keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Softmax
from keras import applications

options = sys.argv[1:]
class_names = os.listdir("data/train")
class_name_indices = {name:i for i,name in enumerate(class_names)}
nbr_classes = len(class_names)

if "-f" in options:
# Calculate the MobileNet's convolution output for each sample and store it.
# This makes transfer learning much faster.
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


def get_generator(df,y_name,batch_size,shuffle=True):
# This generates batches of training or validation data for training the student.    
    X = np.array([s["student_feature_path"] for s in df])
    Y = np.array([s[y_name] for s in df])
    nbr_samples = len(X)
    nbr_batches = nbr_samples // batch_size
        
    while True:
        perm = np.arange(nbr_samples)
        if shuffle:
            np.random.shuffle(perm)
            
        for i in range(nbr_batches):
            indices = perm[i*batch_size:(i+1)*batch_size]
            X_batch = X[indices]
            Y_batch = Y[indices]
            X_batch = np.array([np.load(p) for p in X_batch])
            if y_name=="label":
                Y_batch = keras.utils.to_categorical(Y_batch,nbr_classes)
            yield (X_batch,Y_batch)

if "-s" in options:
# s stands for student (as opposed to train delinquent).  
# This block trains the student using stored convolution features   
    df_train = pickle.load(open("train_data.p","rb"))
    df_test = pickle.load(open("test_data.p","rb"))
    g_train = get_generator(df_train,"teacher_logits",32)
    g_test = get_generator(df_test,"label",32,shuffle=False)
    
    top_model = keras.models.Sequential()
    top_model.add(Flatten(input_shape=(4,4,256)))
    top_model.add(Dense(200,activation="relu"))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(nbr_classes,activation=None,name="logits"))
    top_model.add(Softmax())
    top_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["acc"])
    logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)
    logits_model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer = keras.optimizers.Adam(lr=1e-5))
    
    log = []
    for epoch in range(300):
        logits_model.fit_generator(g_train,steps_per_epoch=len(df_train)//32,epochs=1,verbose=0)
        how_good = top_model.evaluate_generator(g_test,steps=len(df_test)//32)
        print("Epoch {} validation results are {}".format(epoch,how_good))
        log.append(how_good)
    pickle.dump(log,open("stats/student_train_log.p","wb"))
    top_model.save("models/student.h5")

if "-d" in options:
# Train a delinquent model (one that learns from the ground truth labels, ignoring the teacher)
# Used for benchmarking the performance of the student.
    df_train = pickle.load(open("train_data.p","rb"))
    df_test = pickle.load(open("test_data.p","rb"))
    g_train = get_generator(df_train,"label",32)
    g_test = get_generator(df_test,"label",32,shuffle=False)
    
    top_model = keras.models.Sequential()
    top_model.add(Flatten(input_shape=(4,4,256)))
    top_model.add(Dense(200,activation="relu"))
    top_model.add(Dropout(0.3))
    top_model.add(Dense(nbr_classes,activation=None,name="logits"))
    top_model.add(Softmax())
    
    top_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer = keras.optimizers.Adam(lr=1e-5),
        metrics=["acc"])
    results = top_model.fit_generator(g_train,steps_per_epoch=len(df_train)//32,validation_data=g_test,validation_steps=len(df_test)//32,epochs=150)
    pickle.dump(results.history["val_acc"],open("stats/delinquent_val_acc.p","wb"))

