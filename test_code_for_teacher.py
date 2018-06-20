'''
In this file I test whether the saved logits output by Teacher.py 
are consistent with the teacher model applied to a given input image
'''
import keras
import numpy as np
from keras.layers import Input
from keras import applications
import pickle


df = pickle.load(open("../train_data.p","rb"))

conv_model = applications.InceptionV3(include_top=False, weights='imagenet')
top_model = keras.models.load_model("../models/teacher_top_model.h5")
inp = Input(shape=(224,224,3))
x=conv_model(inp)
out = top_model(x)
full_model = keras.models.Model(inputs=inp,outputs=out)
    
def softmax(x):
    p=np.exp(x)
    return p/sum(p)

while input("c to continue testing samples ")=="c":
    sample = np.random.choice(df)
    
    img_path = "../"+sample["image_path"]
    img =keras.preprocessing.image.load_img(img_path,target_size=(224,224))
    img = keras.preprocessing.image.img_to_array(img)
    img = keras.applications.inception_v3.preprocess_input(np.array([img]))
    
    features1 = conv_model.predict(img)[0]
    features2 = np.load("../"+sample["teacher_feature_path"])
    print("The following features should be equal:\n {}\n{}".format(features1[:100],features2[:100]))

    logits_model = keras.models.Model(inputs=top_model.layers[0].input,outputs=top_model.get_layer("logits").output)
    features2 = np.array([features2])
    logits1 = logits_model.predict(features2)[0]
    preds1 = top_model.predict(features2)[0]

    print("Putting features through top model gives the following \nlogits {}\n and probabilities {}".format(logits1,preds1))

    prediction=full_model.predict(img)[0]
    print("Prediction is {}".format(prediction))
    
    print("The stored teacher logits are {}".format(sample["teacher_logits"]))
    compare_prediction_to = softmax(sample["teacher_logits"])
    print("If all is well, the prediction should equal {}".format(compare_prediction_to))


