import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
# example of training a final classification model

traffic_signs = {
    0	: "Speed limit (20km/h)", 
1	:"Speed limit (30km/h)",
2	:"Speed limit (50km/h)",
3	:"Speed limit (60km/h)",
4	:"Speed limit (70km/h)",
5	:"Speed limit (80km/h)",
6	:"End of speed limit (80km/h)",
7	:"Speed limit (100km/h)",
8	:"Speed limit (120km/h)",
9	:"No passing",
10	:"No passing for vechiles over 3.5 metric tons",
11	:"Right-of-way at the next intersection",
12	:"Priority road",
13	:"Yield",
14	:"Stop",
15	:"No vechiles",
16	:"Vechiles over 3.5 metric tons prohibited",
17	:"No entry",
18	:"General caution",
19	:"Dangerous curve to the left",
20	:"Dangerous curve to the right",
21	:"Double curve",
22	:"Bumpy road",
23	:"Slippery road",
24	:"Road narrows on the right",
25	:"Road work",
26	:"Traffic signals",
27	:"Pedestrians",
28	:"Children crossing",
29	:"Bicycles crossing",
30	:"Beware of ice/snow",
31	:"Wild animals crossing",
32	:"End of all speed and passing limits",
33	:"Turn right ahead",
34	:"Turn left ahead",
35	:"Ahead only",
36	:"Go straight or right",
37	:"Go straight or left",
38	:"Keep right",
39	:"Keep left",
40	:"Roundabout mandatory",
41	:"End of no passing",
42	:"End of no passing by vechiles over 3.5 metric tons"
}

model = tf.keras.models.load_model('model_t.h5')
image = cv2.imread('traffic.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.resize(image, (32, 32))

predicition = model.predict_classes(image.reshape(1, 32, 32, 1))


print(traffic_signs[predicition[0]])