import numpy as np 
import matplotlib.pyplot as plt 
import keras 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator 
import pickle 
import pandas as pd 
import random 
import cv2 
np.random.seed(0)

#load the dataset
data = pd.read_csv("C:\\Users\\aishw\\OneDrive\\Documents\\Traffic sign classification\\german-traffic-signs\\signnames.csv")
with open('C:\\Users\\aishw\\OneDrive\\Documents\\Traffic sign classification\\german-traffic-signs\\train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('C:\\Users\\aishw\\OneDrive\\Documents\\Traffic sign classification\\german-traffic-signs\\valid.p', 'rb') as f:
    val_data = pickle.load(f)
with open('C:\\Users\\aishw\\OneDrive\\Documents\\Traffic sign classification\\german-traffic-signs\\/test.p', 'rb') as f:
    test_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

print(X_train)

#preprocess the data
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255 
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)



#image data generator is used to imrpove the training dataset
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,shear_range=0.1, rotation_range=10)
datagen.fit(X_train)
y_train = to_categorical(y_train, 43) 
y_val = to_categorical(y_val, 43) 
y_test = to_categorical(y_test, 43)

#build the model
num_classes = 43
def cnn_model():
    model = Sequential()

    model.add(Conv2D(60, (5,5), input_shape = (32, 32,1), activation = 'relu'))
    model.add(Conv2D(60, (5,5), activation = 'relu'))

    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))


    model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = cnn_model() 
history = model.fit(X_train, y_train,batch_size = 50, steps_per_epoch = 1000, epochs = 10)
model.save('model_t.h5')

#a random image to test
score = model.evaluate(X_test, y_test, verbose = 0)


plt.imshow(X_test[990].reshape(32, 32)) 
print("Predicted sign: "+ str(model.predict(X_test[990].reshape(1, 32, 32, 1)))) 
    



