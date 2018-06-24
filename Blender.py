import numpy as np
import pickle
import keras


def normalize(x):
    m = np.mean(x)
    s = np.std(x)
    return (x-m)/s


models = ["xception","vgg","resnet","inception_resnet","densenet","inception"]
m= len(models)

df_train = pickle.load(open("train_data.p","rb"))
X_train = [[normalize(s["teacher_logits_{}".format(model)]) for model in models] for s in df_train]
X_train = np.array(X_train).transpose(0,2,1)
y_train = np.array([s["label"] for s in df_train])
y_train = keras.utils.to_categorical(y_train)

df_test = pickle.load(open("test_data.p","rb"))
X_test = [[normalize(s["teacher_logits_{}".format(model)]) for model in models] for s in df_test]
X_test = np.array(X_test).transpose(0,2,1)
y_test = np.array([s["label"] for s in df_test])
y_test = keras.utils.to_categorical(y_test)

print("Shapes: X_train {}, X_test {}, y_train {}, y_test {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))

m=len(models)
s=len(df_train)
c=5
blender = keras.models.Sequential()
blender.add(keras.layers.Reshape((5,1,m),input_shape=(5,m)))
blender.add(keras.layers.Conv2D(1,(1,1),name = "conv"))
blender.add(keras.layers.Reshape((5,),name= "logits"))
blender.add(keras.layers.Softmax())
blender.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(),
    metrics = ["acc"])

#wts = [np.array([[[[1.0],[0.3],[0.5],[1.0]]]]),np.array([0])]
#blender.get_layer("conv").set_weights(wts)

blender.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),batch_size=200)
print(blender.get_layer("conv").get_weights())

logit_model = keras.models.Model(inputs=blender.layers[0].input,outputs=blender.get_layer("logits").output)
logits = logit_model.predict(X_train)
for s,l in zip(df_train,logits):
    s["blended_logits"] = l
pickle.dump(df_train,open("train_data.p","wb"))









