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
#class_names = os.listdir("data/train") # Train should have a subfolder for each class like "cat", "dog", etc.
class_names= ["cat","dog","bear"]
class_name_indices = {name:i for i,name in enumerate(class_names)}
img_width, img_height = 224, 224

model_name = "densenet"


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



def build_database():
# Build a database containing the location of each image, and the image's label
# The database will be a list of dictionaries
# Separate databases for train and test
    for case in ["train","test"]: 
        database = []
        for class_name in class_names:
            source_folder = "data/{}/{}".format(case,class_name)
            for filename in os.listdir(source_folder):
                img_path="{}/{}".format(source_folder,filename)
                database.append({
                    "image_path":img_path,
                    "label":class_name_indices[class_name],
                    "label_name":class_name})
        pickle.dump(database,open("{}_data.p".format(case),"wb"))

class MySaver(keras.callbacks.Callback):
    def __init__(self):
        self.ckpt_at = [.70,.75,.80,.85,.86,.87,.88,.89,.895,.900,.905,.910,.915,.920,.925,0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01]
        
    def on_epoch_end(self,epoch,logs={"val_acc":0.0}):
        acc = logs["val_acc"]
        if acc>self.ckpt_at[0]:
            self.ckpt_at = [x for x in self.ckpt_at if x>acc]
            self.model.save("teacher_{}_{}_{:.5f}.h5".format(model_name,epoch,acc))
 
class Teacher:

    def __init__(self):
        return

    def calculate_features(self):
    # Calculate output of convolutional layers for each sample, and saves them so that transfer learning can be done quickly
        def preprocess(p):  #preprocess the image stored at path p
            img = keras.preprocessing.image.load_img(p,target_size=(img_width,img_height))
            img = keras.preprocessing.image.img_to_array(img)
            img = preprocessors[model_name](np.array([img]))
            return img[0]
    
        conv_model = model_constructors[model_name](include_top=False, weights='imagenet')
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
    
    
           #else:
            #    print(acc,self.ckpt_at)
    def train_teacher(self):
        batch_size = 32
        df_train = pickle.load(open("train_data.p","rb"))
        df_test = pickle.load(open("test_data.p","rb"))
        features_shape = np.load(df_train[0]["teacher_feature_path"]).shape
        nbr_train = len(df_train)
        nbr_test = len(df_test)
        saver = MySaver()
    
        train_generator = Teacher.batch_generator(df_train,batch_size)
        test_generator = Teacher.batch_generator(df_test,batch_size,shuffle=False)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=features_shape))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(len(class_names), activation=None,name="logits"))
        top_model.add(Softmax())
        top_model.compile(
            optimizer=keras.optimizers.RMSprop(lr=1e-6),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        ckpt = keras.callbacks.ModelCheckpoint("models/teacher_{}.h5".format(model_name),monitor='val_acc',save_best_only=1)
        h = top_model.fit_generator(
            train_generator,
            steps_per_epoch=nbr_train // batch_size,
            validation_data=test_generator,
            validation_steps=nbr_test // batch_size,
            epochs=100)
            #callbacks=[ckpt,saver])
        print("The keys are {}".format(h.history.keys()))
        #pickle.dump(h.history["val_acc"],open("stats/teacher_val_acc.p","wb"))
        top_model.save("models/teacher_top_model.h5")
        return h
    
                   
    def store_logits(self):
    # Calculate the rich labels,
    # which for now are the teacher model's logits (hidden state before final softmax)
    # Saves the rich labels so that the student can be trained with them.
        #top_model = keras.models.load_model("models/teacher_{}.h5".format(model_name))
        top_model = keras.models.load_model("models/teacher_top_model.h5")
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
                sample["teacher_logits".format(model_name)]=logits
            pickle.dump(df,open("{}_data.p".format(case),"wb"))
    



teacher1 = Teacher()
if "-d" in options:
    build_database()

if "-f" in options:
    teacher1.calculate_features()

if "-t" in options:
    teacher1.train_teacher()
if "-l" in options:
    teacher1.store_logits()


