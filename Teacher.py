'''
In this script we build the teacher model used in distillation,
and also use it to generate the rich labels.
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input, Softmax
from keras import applications
import re, os, sys, pickle, keras

options = sys.argv[1:]
class_names = os.listdir("data/train") # Train should have a subfolder for each class like "cat", "dog", etc.
#class_names= ["cat","dog","bear","horse","squirrel"]
class_name_indices = {name:i for i,name in enumerate(class_names)}
img_width, img_height = 224, 224

model_constructors = {
    "inception":applications.inception_v3.InceptionV3,
    "xception":applications.xception.Xception,
    "vgg":applications.vgg19.VGG19,
    "resnet":applications.resnet50.ResNet50,
    "inception_resnet":applications.inception_resnet_v2.InceptionResNetV2,
    "densenet":applications.densenet.DenseNet201}
preprocessors = {
    "inception":applications.inception_v3.preprocess_input,
    "xception":applications.xception.preprocess_input,
    "vgg":applications.vgg19.preprocess_input,
    "resnet":applications.resnet50.preprocess_input,
    "inception_resnet":applications.inception_resnet_v2.preprocess_input,
    "densenet":applications.densenet.preprocess_input}   




class Teacher:

    def __init__(self,model_name="densenet"):
        self.model_name = model_name

    def calculate_features(self):
    # Calculate output of convolutional layers for each sample, and saves them so that transfer learning can be done quickly
        def preprocess(p):  #preprocess the image stored at path p
            img = keras.preprocessing.image.load_img(p,target_size=(img_width,img_height))
            img = keras.preprocessing.image.img_to_array(img)
            img = preprocessors[self.model_name](np.array([img]))
            return img[0]
    
        conv_model = model_constructors[self.model_name](include_top=False, weights='imagenet')
        for case in ["train","test"]:
            print(case)
            df = pickle.load(open("{}_data.p".format(case),"rb"))
            l=len(df) // 10
            for i,sample in enumerate(df):
                if i%l==0:
                    print(i//l)
                name = sample["label_name"]
                path = sample["image_path"]
                img = preprocess(path)
                features = conv_model.predict(img.reshape(1,*img.shape))[0]
                new_path = re.sub("data","features/teacher",path)
                new_path = re.sub(r".jpg$",".npy",new_path)
                sample["teacher_feature_path"]=new_path
                np.save(new_path,features)
            pickle.dump(df,open("{}_data.p".format(case),"wb"))
        
       
    @staticmethod
    def batch_generator(df,batch_size,shuffle=True):
    # Generates train and validation batches for training the teacher model
        X = np.array([s["teacher_feature_path"] for s in df])
        Y = np.array([s["label"] for s in df])
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
                Y_batch = keras.utils.to_categorical(Y_batch,num_classes=len(class_names))
                yield (X_batch,Y_batch)
    
    def train(self,epochs=100,lr=1e-5,batch_size=32):
        df_train = pickle.load(open("train_data.p","rb"))
        df_test = pickle.load(open("test_data.p","rb"))
        nbr_train = len(df_train)
        nbr_test = len(df_test)
        train_generator = Teacher.batch_generator(df_train,batch_size)
        test_generator = Teacher.batch_generator(df_test,batch_size,shuffle=False)

        features_shape = np.load(df_train[0]["teacher_feature_path"]).shape
        top_model = Sequential()
        top_model.add(Flatten(input_shape=features_shape))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(len(class_names), activation=None,name="logits"))
        top_model.add(Softmax())
        top_model.compile(
            optimizer=keras.optimizers.RMSprop(lr=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        ckpt = keras.callbacks.ModelCheckpoint("models/teacher_{}.h5".format(self.model_name),monitor='val_acc',save_best_only=1)
        h = top_model.fit_generator(
            train_generator,
            steps_per_epoch=nbr_train // batch_size,
            validation_data=test_generator,
            validation_steps=nbr_test // batch_size,
            epochs=epochs,
            callbacks=[ckpt])
        return h
    
                   
    def store_logits(self):
    # Calculate the rich labels,
    # which for now are the teacher model's logits (hidden state before final softmax)
    # Saves the rich labels so that the student can be trained with them.
        top_model = keras.models.load_model("models/teacher_{}.h5".format(self.model_name))
        logits_model = keras.models.Model( # build a model that grabs the hidden state just before final softmax
            inputs=top_model.layers[0].input,
            outputs=top_model.get_layer("logits").output)
        for case in ["train","test"]:
            df = pickle.load(open("{}_data.p".format(case),"rb"))
            for sample in df: # calculate logits for each training sample
                feature_path = sample["teacher_feature_path"]
                features = np.load(feature_path)
                features = np.array([features])
                logits = logits_model.predict(features)[0]
                logits -= np.mean(logits)
                sample["teacher_logits".format(self.model_name)]=logits
            pickle.dump(df,open("{}_data.p".format(case),"wb"))
    



