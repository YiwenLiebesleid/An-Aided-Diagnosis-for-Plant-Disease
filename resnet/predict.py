import numpy as np
import os
from PIL import Image
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *gyw
from keras.models import load_model

file_path = 'testimg.jpg'
flag = 1	#1 to reshape

def reshape_pic():
	im = Image.open(file_path)
	im = im.resize((224,224))
	return im
	
def load_label():
	labels = []
	path = 'resnet/label.txt'
	file = open(path)
	while 1:
		line = file.readline()
		if not line:
			break
		else:
			labels.append(line)
	return labels

def decode(y,top):
	labels = load_label()
	lenth = labels.__len__()
	if lenth < top:
		top = lenth
	y = np.array(y)[0]
	rank = np.argsort(-y)
	ret_msg = []
	ret_prob = []
	for r in range(top):
		ret_msg.append(labels[rank[r]].split('\n')[0])
		ret_prob.append(y[rank[r]])
	return ret_msg, ret_prob

if __name__ == '__main__':
	if flag == 0:
		img = image.load_img(file_path, target_size=(224, 224))
	else:
		img = reshape_pic()
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	model = load_model("test.h5")
	y = model.predict(x)
	print('\nPredicted:', decode(y,top=3))
	ret_msg, ret_prob = decode(y,top=3)
	for i in range(len(ret_msg)):
		print(str(i)+": "+ret_msg[i],ret_prob[i])
	# print(y)
