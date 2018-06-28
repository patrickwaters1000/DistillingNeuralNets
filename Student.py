import os, sys, re, pickle, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Softmax, concatenate
import keras.backend as K

class_names = os.listdir("data/train")
#class_names = ["cat","dog","bear","horse","squirrel"]
class_name_indices = {name:i for i,name in enumerate(class_names)}
nbr_classes = len(class_names)


def get_features(img_path,model):
    img = keras.preprocessing.image.load_img(img_path,target_size=(128,128))
    img = keras.preprocessing.image.img_to_array(img)
    img = keras.applications.mobilenet.preprocess_input(np.array([img]))
    return model.predict(img)[0]

def distillation_loss(c=0.0,T=1.0):
    # y_true is a concatenation of the teacher's logits and the true labels
    # z_ is the student's logits
    def f(y_true,z_):
        z,labels = y_true[:,0:nbr_classes],y_true[:,nbr_classes:]
        xent_loss = K.categorical_crossentropy(labels,z_,from_logits=True)
        dist_loss = K.categorical_crossentropy(K.softmax(z/T),z_/T,from_logits=True)
        return dist_loss + c*xent_loss
    return f

def distillation_loss_mse(c=0.0):
    def f(y_true,z_):
        z,labels = y_true[:,0:nbr_classes],y_true[:,nbr_classes:]
        xent_loss = K.categorical_crossentropy(labels,z_,from_logits=True)
        dist_loss = K.sum((z-z_)**2)
        return dist_loss + c*xent_loss
    return f

class Student:
    def __init__(self):
        return

    def calculate_features(self):
        convnet = keras.applications.MobileNet(
            input_shape=(128,128,3),
            alpha=0.25,
            include_top=False,
            weights="imagenet")
        for case in ["train","test"]:
            df = pickle.load(open("{}_data.p".format(case),"rb"))
            for sample in df:
                img_path = sample["image_path"]
                feature_path = re.sub(r".jpg$",".npy",img_path)
                feature_path = re.sub(r"^data","features/student",feature_path)
                features = get_features(img_path,convnet)
                np.save(feature_path,features)
                sample["student_feature_path"]=feature_path
            pickle.dump(df,open("{}_data.p".format(case),"wb"))

    @staticmethod
    def batch_generator(df,y_name,batch_size=32,shuffle=True):
    # generates test data batches, or training batches for the delinquent
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
    def distill_train_generator(df,batch_size=32,shuffle=True):
    # generates minibatches for distillation training
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


    def train_student(self,c=0.0,T=1.0,use_mse=False,epochs=200,lr=1e-5,batch_size=32):
        if use_mse:
            loss = distillation_loss_mse(c=c)
        else:
            loss = distillation_loss(c=c,T=T)
        df_train = pickle.load(open("train_data.p","rb"))
        df_test = pickle.load(open("test_data.p","rb"))
        g_train = Student.distill_train_generator(df_train,batch_size=batch_size)
        g_test = Student.batch_generator(df_test,"label",batch_size=batch_size,shuffle=False)
        
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
   
        logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)
        logits_model.compile(loss=loss,optimizer = keras.optimizers.Adam(lr=lr))
        
        log = []
        for epoch in range(200):
            logits_model.fit_generator(g_train,steps_per_epoch=len(df_train)//batch_size,epochs=1,verbose=0)
            how_good = top_model.evaluate_generator(g_test,steps=len(df_test)//batch_size)
            print("Epoch {} validation results are {}".format(epoch,how_good))
            log.append(how_good)

        top_model.save("models/student.h5")
        losses,val_accs = zip(*log)
        print("Best ={}".format(max(val_accs)))
        return val_accs

    def train_delinquent(self,epochs=200,lr=1e-5,batch_size=32):
        df_train = pickle.load(open("train_data.p","rb"))
        df_test = pickle.load(open("test_data.p","rb"))
        g_train = Student.batch_generator(df_train,"label",batch_size=batch_size)
        g_test = Student.batch_generator(df_test,"label",batch_size=batch_size,shuffle=False)
        
        top_model = keras.models.Sequential()
        top_model.add(Flatten(input_shape=(4,4,256)))
        top_model.add(Dense(200,activation="relu"))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(nbr_classes,activation=None,name="logits"))
        top_model.add(Softmax())
        
        top_model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer = keras.optimizers.Adam(lr=lr),
            metrics=["acc"])
        results = top_model.fit_generator(
            g_train,
            steps_per_epoch=len(df_train)//batch_size,
            validation_data=g_test,
            validation_steps=len(df_test)//batch_size,
            epochs=epochs)
        return(results)

