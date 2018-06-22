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
class_name_indices = {name:i for i,name in enumerate(class_names)}
img_width, img_height = 224, 224

if "-d" in options: 
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
                    "label_name":class_name,
        pickle.dump(database,open("{}_data.p".format(case),"wb"))


if "-f" in options:
# Calculate output of convolutional layers for each sample, and saves them so that transfer learning can be done quickly
    def preprocess(p):  #preprocess the image stored at path p
        img = keras.preprocessing.image.load_img(img_path,target_size=(img_width,img_height))
        img = keras.preprocessing.image.img_to_array(img)
        img = keras.applications.inception_v3.preprocess_input(np.array([img]))
        return img[0]

    conv_model = applications.InceptionV3(include_top=False, weights='imagenet')
    for case in ["train","test"]:
        df = pickle.load(open("{}_data.p".format(case),"rb"))
	for sample in df:
            name = sample["label_name"]
            path = sample["image_path"]
            img = preprocess(path)
            features = conv_model.predict(img.reshape(1,*img.shape))[0]
            new_path = re.sub("data","features/teacher",path)
            new_path = re.sub(r".jpg$",".npy",path)
            sample["teacher_feature_path"]=new_path
            np.save(new_path,features)
        pickle.dump(df,open("{}_data.p".format(case),"wb"))
    
   

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
            Y_batch = keras.utils.to_categorical(Y_batch,num_classes=5)
            yield (X_batch,Y_batch)



if "-t" in options:
    df_train = pickle.load(open("train_data.p","rb"))
    df_test = pickle.load(open("test_data.p","rb"))
    nbr_train = len(df_train)
    nbr_test = len(df_test)

    feature_shape=(5,5,2048)
    train_generator = batch_generator(df_train,32)
    test_generator = batch_generator(df_test,32,shuffle=False)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=(5,5,2048)))
    top_model.add(Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(len(class_names), activation=None,kernel_regularizer=keras.regularizers.l2(0.0001),name="logits"))
    top_model.add(Softmax())
    top_model.compile(
        optimizer=keras.optimizers.RMSprop(lr=1e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    ckpt = keras.callbacks.ModelCheckpoint("models/teacher_epoch{epoch:03d}.h5")
    h = top_model.fit_generator(
        train_generator,
        steps_per_epoch=nbr_train // 32,
        validation_data=test_generator,
        validation_steps=nbr_test // 32,
        epochs=40,
        callbacks=[ckpt])
    print("The keys are {}".format(h.history.keys()))
    pickle.dump(h.history["val_acc"],open("stats/teacher_val_acc.p","wb"))
    top_model.save("models/teacher_top_model.h5")


               
if "-l" in options:
# Calculate the rich labels,
# which for now are the teacher model's logits (hidden state before final softmax)
# Saves the rich labels so that the student can be trained with them.
    top_model = keras.models.load_model("models/teacher_top_model.h5")
    logits_model = keras.models.Model( # build a model that grabs the hidden state just before final softmax
        inputs=top_model.layers[0].input,
        outputs=top_model.get_layer("logits").output)

    df = pickle.load(open("train_data.p","rb"))
    for sample in df: # calculate logits for each training sample
        feature_path = sample["teacher_feature_path"]
        features = np.load(feature_path)
        features = np.array([features])
        logits = logits_model.predict(features)[0]
        sample["teacher_logits"]=logits
    pickle.dump(df,open("train_data.p","wb"))

