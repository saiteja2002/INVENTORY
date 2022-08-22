import pickle
import numpy as np
from basemodel_architeture_search import basemodel_architeture_search
from keras_preprocessing.image import ImageDataGenerator
from SampleSpaceGenerate_CreateandTrainModel import Generate_Sample_Space_and_Create_and_train_model


# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        r"C:\Users\saite\PycharmProjects\NAS\aa\DATASETS\TRAIN__",  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=128,
        # Since you used binary_crossentropy loss, you need binary labels
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        r'C:\Users\saite\PycharmProjects\NAS\aa\DATASETS\TEST__',  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=32,
        # Since you used binary_crossentropy loss, you need binary labels
        class_mode='binary')

main = basemodel_architeture_search(train_generator, validation_generator)
#running the search
data = main.search()
#getting the top 5 architetures for this data
n=6
data_file = 'model_data_generated_sofar.pkl'
with open(data_file, 'rb') as f:
    data = pickle.load(f)
val_accs = [item[1] for item in data]
sorted_idx = np.argsort(val_accs)[::-1]
data = [data[x] for x in sorted_idx]
search_space = Generate_Sample_Space_and_Create_and_train_model()
for seq_data in data[:n]:
    print('Architecture', search_space.id_to_config(seq_data[0]),"        ",'Validation Accuracy:', seq_data[1])
