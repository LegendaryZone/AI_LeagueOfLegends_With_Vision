# x_data, y_data
import tensorflow.keras.backend as K
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from tensorflow.keras import optimizers


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras import callbacks

import time
import os
import numpy as np

from dataset_loader import Loader_nasnet

os.environ['KMP_DUPLICATE_LIB_OK']='True'

tf.enable_eager_execution()


AUTOTUNE = tf.data.experimental.AUTOTUNE #-1

flags = tf.app.flags

flags.DEFINE_string("TB_CP", "./logs/", "tensorboard and checkpoint")

flags.DEFINE_integer("units", 3, "steps")
flags.DEFINE_integer("epochs", 1, "epoch")


flags.DEFINE_integer("batch_size", 1, "the size of batch")
flags.DEFINE_integer("width", 331, "width")
flags.DEFINE_integer("height", 331, "height")


FLAGS = flags.FLAGS


def main(_):

	# Image dataset
	x_data, y_data = Loader_nasnet()

	print(x_data.shape, y_data.shape)

	training_data = tf.data.Dataset.from_tensor_slices((x_data, y_data)).repeat().batch(FLAGS.batch_size)

	mobile_net = tf.keras.applications.nasnet.NASNetLarge(input_shape=(FLAGS.width,FLAGS.height, 3), include_top=False)
	mobile_net.trainable = False


	model = Sequential([
		mobile_net,
		#28 28 ,32 14,14,32 7,7,32
		Conv2D(filters = 32, kernel_size=(3,3), strides= 1,
			padding ="same", activation= "relu"),
		MaxPooling2D(pool_size=(3,3),strides =2, padding = "same"),
		Dropout(rate = 0.5),
		# 4,4,64
		Conv2D(filters =64, kernel_size = (3,3), strides=1,
			padding = "same", activation = "relu"),
		MaxPooling2D(pool_size=(3,3), strides = 2, padding = "same"),
		Dropout(rate=0.5),

		Conv2D(filters = 128, kernel_size=(3,3), strides = 1,
			padding="same", activation="relu"),
		MaxPooling2D(pool_size=(3,3), strides=2, padding= "same"),
		Dropout(rate=0.5),
		# -1 , 4*4*64 625 3
		Flatten(),
		Dense(units= 625, activation = "relu"),
		Dense(units = 125, activation = "relu"),
		Dense(units = 3, activation = "sigmoid")
		])

	# model.compile(loss = "categorical_crossentropy",
	# 	optimizer = "adam",
	# 	metrics = ["accuracy"])
	model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

# model.fit(train_data, train_labels,
#               epochs=epochs,
#               batch_size=batch_size,
#               validation_data=(validation_data, validation_labels))
#     model.save_weights(top_model_weights_path)
	# cb_tb = callbacks.TensorBoard(log_dir = "./logs/", histogram_freq=2)
	# cb_es = callbacks.EarlyStopping(patience =5, monitor= "val_loss")
	# keras_ds = ds.map(change_range)
	# image_batch, label_batch = next(iter(keras_ds))


	# callbacks = [cb_tb, cb_es]
	# steps_per_epoch=tf.ceil(image_count/FLAGS.batch_size).numpy()

	model.fit(training_data,
		epochs = FLAGS.epochs,	
		steps_per_epoch=(len(x_data) // FLAGS.batch_size),
		validation_data = training_data,
		validation_steps=(len(x_data) // FLAGS.batch_size))
	# model.fit(ds, 
	# 	# validation_data = ds,
	# 	epochs=FLAGS.epoch, 
	# 	steps_per_epoch=int(steps_per_epoch)) 
	# 	# callbacks=callbacks,
	# 	# validation_steps = 1) # tensorflow==1.13.0rc


	# model.fit_generator(
	# 	train_generator,
	# 	validation_data= eval_generator,
	# 	epochs= training_epoch,
	# 	callbacks=callbacks,
	# 	validation_steps= 1)
	

	
	# Load model
	# model = tf.keras.models.load_model("model.h5")
	# model.summary()

	model.evaluate(training_data, batch_size=32, steps=1)

	model.save("model.h5")
	model.save_weights("weights.h5")


	# new_model.summary()

	# new_model.evaluate(training_data, batch_size=1, steps=1)

	model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
	prediction = model.predict_classes(x_data)

	# prediction = model.predict(x_data, batch_size=1, steps=1)
	print("prediction", prediction)
	# model.predict_on_batch(x_data)

	model.summary()




	# output_names = [node.op.name for node in model.outputs]
	# input_names = [node.op.name for node in model.inputs]
	# print(input_names)
	# print(output_names)

	# def save_graph_to_file(sess,  graph_file_name, output_names):
	#     output_graph_def = graph_util.convert_variables_to_constants(
	#       sess,  sess.graph.as_graph_def(),  output_names)
	#     with gfile.FastGFile(graph_file_name, 'wb') as f:
	#         f.write(output_graph_def.SerializeToString())

	# export_dir = './weight/'
	# sess = K.get_session()
	# save_graph_to_file(sess,  export_dir + "flower5.pb", output_names)



	"""

	toco \
	  --graph_def_file=weight/flower5.pb \
	  --output_file=weight/flower5.lite \
	  --input_format=TENSORFLOW_GRAPHDEF \
	  --output_format=TFLITE \
	  --input_shape= 28, 28 ,3 \
	  --input_array='conv2d_input' \
	  --output_array='dense_1/Softmax' \
	  --inference_type=FLOAT \
	  --input_data_type=FLOAT
	"""


	# for e in range(training_epoch):
	#     print('Epoch', e)
	#     batches = 0
	#     for x_batch, y_batch in total_datagen.flow(x_train, y_train, batch_size=32):
	#         model.fit(x_batch, y_batch)
	#         batches += 1
	#         if batches >= len(x_train) / 32:
	#             # we need to break the loop by hand because
	#             # the generator loops indefinitely
	#             break







if __name__ == "__main__":
	tf.app.run()



# 