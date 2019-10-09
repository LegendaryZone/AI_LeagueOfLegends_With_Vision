import numpy as np
import tensorflow as tf
import os

from sklearn.preprocessing import OneHotEncoder
from tf_load_image import Load_Image

data = np.load("data/LOL_data.npy")

image_data = []
label_data = []

for x_data, y_data in data:
    image_data.append(x_data)
    label_data.append(y_data)
image_data, label_data = np.array(image_data), np.array(label_data).reshape(-1,1)
image_data = np.array(image_data).reshape(len(image_data[-1,:]))
image_data = np.array([i for i in image_data])
encoder = OneHotEncoder()
encoder.fit(label_data)

label_data = encoder.transform(label_data).toarray()
print(image_data.shape)

date_set =[]
date_set.append([image_data, label_data])

print(date_set)

train_size = round(int(len(date_set) *0.8))
train_data = date_set[0:train_size]
test_data = date_set[train_size:]
# print(train_data)
print(image_data)
print(label_data)