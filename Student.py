
import os
import keras
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Softmax, concatenate
from keras import applications
import keras.backend as K
import re
import pickle

options = sys.argv[1:]
#class_names = os.listdir("data/train")
class_names = ["cat","dog","bear"]
class_name_indices = {name:i for i,name in enumerate(class_names)}
nbr_classes = len(class_names)
print("class names are {}".format(class_names))

batch_size=32

student_convnet = keras.applications.MobileNet(input_shape=(128,128,3),alpha=0.25,include_top=False,weights="imagenet")
def get_features(img_path):
    img = keras.preprocessing.image.load_img(img_path,target_size=(128,128))
    img = keras.preprocessing.image.img_to_array(img)
    img = keras.applications.mobilenet.preprocess_input(np.array([img]))
    return student_convnet.predict(img)[0]

def distillation_loss(c,a):
    def f(y_true,y_pred):
        pred_logits = a * y_pred[:,0:nbr_classes]
        pred_probs = y_pred[:,nbr_classes:]
        teach_logits = y_true[:,0:nbr_classes]
        true_label = y_true[:,nbr_classes:]
        loss1 = K.sum((pred_logits - teach_logits)**2)
        m=K.max(pred_logits,axis=-1)
        m=K.reshape(m,(batch_size,1))
        loss2 = - K.sum( (pred_logits - m)*true_label)
        return loss1 + c*loss2
    return f

class Student:
    def __init__(self):
        return

    

    def calculate_features(self):
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

    @staticmethod
    def get_generator(df,y_name,batch_size,shuffle=True):
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

    @staticmethod
    def special_generator(df,batch_size,shuffle=True):
        X = np.array([s["student_feature_path"] for s in df])
        Y1 = np.array([s["teacher_logits"] for s in df])
        Y2 = np.array([s["label"] for s in df])
        Y2 = keras.utils.to_categorical(Y2)
        Y = np.array([ list(y1)+list(y2) for y1,y2 in zip(Y1,Y2)])
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
                yield (X_batch,Y_batch)


    def train_student(self,a,c):
        df_train = pickle.load(open("train_data.p","rb"))
        df_test = pickle.load(open("test_data.p","rb"))
        g_train = Student.special_generator(df_train,batch_size)
        g_test = Student.get_generator(df_test,"label",batch_size,shuffle=False)
        
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
        x = top_model.get_layer("logits").output
        y = Softmax()(x)
        z = concatenate([x,y],axis=-1)
        model2 = keras.models.Model(inputs=top_model.layers[0].input,outputs=z)
        model2.compile(
            loss = distillation_loss(c,a),
            optimizer = keras.optimizers.Adam(1e-5))
    
        logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)
        logits_model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer = keras.optimizers.Adam(lr=1e-5))
        
        log = []
        best= 0.0
        for epoch in range(200):
            model2.fit_generator(g_train,steps_per_epoch=len(df_train)//batch_size,epochs=1,verbose=0)
            how_good = top_model.evaluate_generator(g_test,steps=len(df_test)//batch_size)
            print("Epoch {} validation results are {}".format(epoch,how_good))
            log.append(how_good)
            acc = how_good[1]
            if acc>best:
                best = acc
        #pickle.dump(log,open("stats/student_train_log.p","wb"))
        top_model.save("models/student.h5")
        print("Best ={}".format(best))
        return best

    def train_delinquent(self):
        df_train = pickle.load(open("train_data.p","rb"))
        df_test = pickle.load(open("test_data.p","rb"))
        g_train = Student.get_generator(df_train,"label",32)
        g_test = Student.get_generator(df_test,"label",32,shuffle=False)
        
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
        results = top_model.fit_generator(g_train,steps_per_epoch=len(df_train)//32,validation_data=g_test,validation_steps=len(df_test)//32,epochs=200)
        #pickle.dump(results.history["val_acc"],open("stats/delinquent_val_acc.p","wb"))
        return(results)

