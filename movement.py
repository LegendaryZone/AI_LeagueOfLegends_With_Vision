from tensorflow.keras.models import load_model
from dataset_loader import Loader_nasnet
import tensorflow as tf

x_data, y_data = Loader_nasnet()

print(x_data.shape, y_data.shape)

training_data = tf.data.Dataset.from_tensor_slices((x_data, y_data)).repeat().batch(1)
test_data = tf.data.Dataset.from_tensor_slices((x_data)).repeat().batch(1)

new_model = load_model("model.h5")


new_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

# new_model.summary()

# new_model.evaluate(training_data, batch_size=1, steps=1)

# new_model.predict(test_data, batch_size=1, steps=1)



