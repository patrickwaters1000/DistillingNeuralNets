from keras.applications import inception_v3
import numpy as np
import os, sys

options = sys.argv[1:]

def soften(x,f):
	'multiplies temperature of prob dist x by factor f'
	y=[np.exp(np.log(xi)/f) for xi in x]
	s=sum(y)
	return np.array([yi/s for yi in y])

x_train = np.load("Data/x_train1_inception.npy")
x_test = np.load("Data/x_test_inception.npy")
y_train = np.load("Data/y_train1.npy")
y_test = np.load("Data/y_test.npy")

print("Shapes")
for a in [x_train,x_test,y_train,y_test]:
	print(a.shape)

print("Building model")
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

animals = open("animals","r").read().split("\n")[:5]
nbr_classes = len(animals)


model = Sequential()
#model.add(Dropout(0.5,input_dim=x_train.shape[1]))
#model.add(Dense(100, activation="relu",input_dim=x_train.shape[1]))
#model.add(Dropout(0.5))
model.add(Dense(nbr_classes, activation="softmax",input_dim=x_train.shape[1]))
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
if not os.path.exists("Weights"):
	os.makedirs("Weights")
if "-t" in options:
	model.fit(	x_train,y_train,
			epochs=50,batch_size=5,
			validation_data=(x_test,y_test)		)
	model.save_weights("Weights/Inception.h5")
else:
	model.load_weights("Weights/Inception.h5")


x_for_rich_labels = np.load("Data/x_train2_inception.npy")
rich_labels = model.predict(x_for_rich_labels)

rich_labels = soften(rich_labels,2)
if not os.path.exists("RichLabels"):
	os.makedirs("RichLabels")
np.save("RichLabels/rich_labels1.npy",rich_labels)


