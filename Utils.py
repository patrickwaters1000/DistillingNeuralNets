import os, pickle, keras

class_names = os.listdir("data/train")
class_name_indices = {name:i for i,name in enumerate(class_names)}

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
 
