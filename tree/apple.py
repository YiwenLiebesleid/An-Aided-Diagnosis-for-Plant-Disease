"""
	This file is for apple nodes
"""
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm import tqdm
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation,Flatten

from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input

from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD

from keras.models import load_model
import numpy as np
import datagen as dg


batch_size = 64
num_classes = 4
epochs = 10
data_augmentation = True

#all_size = 15855
all_size = 15000
epoch_size = 3000
valid_size = int((epoch_size // batch_size) * batch_size)
times = valid_size // batch_size

def batch_gen(filelist):
	train_data = []
	train_label = []
	test_data = []
	test_label = []
	for t in tqdm(range(times)):
		train_batch, train_batch_label, test_batch, test_batch_label, fileleft = dg.image_generator(filelist,batch_size,0.8)
		train_data.append(train_batch)
		train_label.append(train_batch_label)
		test_data.append(test_batch)
		test_label.append(test_batch_label)
		filelist = fileleft

	train_data = np.vstack(train_data)
	train_label = np.vstack(train_label)
	test_data = np.vstack(test_data)
	test_label = np.vstack(test_label)

	return train_data, train_label, test_data, test_label, filelist


#train_data, train_label, test_data, test_label = dg.image_generator(filelist,train_num)

#train_data, train_label, test_data, test_label = dg.image_generator_test(filelist,1000,0.8)

def resnet_model(train_data,train_label,test_data,test_label,flag):
	# model
	input_tensor = Input(shape=(224, 224, 3))
	base_model = ResNet50(input_tensor = input_tensor,weights='imagenet', include_top=False)
	x = base_model.output
	#x = AveragePooling2D()(x)
	x = Flatten()(x)
	# 添加一个全连接层
	x = Dense(1024, activation='relu')(x)

	# 添加一个分类器，假设我们有200个类
	predictions = Dense(4, activation='softmax')(x)
	# 构建我们需要训练的完整模型
	if flag == 0:
		#model = Model(inputs=input_tensor, outputs=predictions)

		#model.compile(optimizer=SGD(lr=0.001,momentum=0.9),loss='categorical_crossentropy',metrics=['acc'])
		
		model = load_model("tree/model/apple_413.h5")
		
	else:
		model = load_model("tree/model/apple_413.h5")


	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(train_data,train_label,batch_size=batch_size,
				  epochs=epochs,validation_data=(test_data,test_label),
				  shuffle=True)
	else:
		print('Using real-time data augmentation.')
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			zca_epsilon=1e-06,  # epsilon for ZCA whitening
			rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
			# randomly shift images horizontally (fraction of total width)
			width_shift_range=0.1,
			# randomly shift images vertically (fraction of total height)
			height_shift_range=0.1,
			shear_range=0.,  # set range for random shear
			zoom_range=[0.8,1.2],  # set range for random zoom
			channel_shift_range=0.,  # set range for random channel shifts
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			cval=0.,  # value used for fill_mode = "constant"
			horizontal_flip=True,  # randomly flip images
			vertical_flip=True,  # randomly flip images
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0
		)

		datagen.fit(train_data)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(datagen.flow(train_data, train_label,
										 batch_size=batch_size),
							epochs=epochs,
							validation_data=(test_data, test_label),
							workers=4)

	model.save("tree/model/apple_413.h5")

if __name__=='__main__':
	filelist = dg.file_gen("./dataset1",18,22)
	total_times = all_size // epoch_size
	for t in range(total_times):
		train_data, train_label, test_data, test_label, filelist = batch_gen(filelist)
		resnet_model(train_data,train_label,test_data,test_label,t)