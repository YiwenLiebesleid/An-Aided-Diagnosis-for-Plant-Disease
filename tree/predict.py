import numpy as np
import os
from PIL import Image
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
from keras.models import load_model

file_path = '../dataset1/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/cut/0-100.JPG'
flag = 0	#1 to reshape

def reshape_pic():
	im = Image.open(file_path)
	im = im.resize((224,224))
	return im
	
def load_label(layer,pred):
	labels = []
	path = ''
	if layer == 0:	#root
		path = 'label/label_all.txt'
	else:	#leaf
		if pred == 'corn':
			path = 'label/label_corn.txt'
		elif pred == 'grape':
			path = 'label/label_grape.txt'
		elif pred == 'tomato':
			path = 'label/label_tomato.txt'
		elif pred == 'apple':
			path = 'label/label_apple.txt'
		else:
			# do nothing
			path = 'label/label_all.txt'
	#print(str(layer) + ":" + path)
	doc = open(path)
	while 1:
		line = doc.readline()
		if not line:
			break
		else:
			labels.append(line)
	return labels

def decode(y,top,layer,pred=''):
	labels = load_label(layer,pred)
	lenth = labels.__len__()
	if lenth < top:	# set the number of top N
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

	#root
	model1 = load_model("model/all_415.h5")
	y1 = model1.predict(x)
	print('\nRoot predict:', decode(y1,top=1,layer=0))
	ret_msg, ret_prob = decode(y1,top=1,layer=0)
	pred_cata = ret_msg[0]
	pred_cata = str.lower(pred_cata)
	
	#leaf
	model2 = load_model("model/"+pred_cata+"_413.h5")
	y2 = model2.predict(x)
	print("\n\t" + pred_cata + ' leaf predict:')
	ret_msg, ret_prob = decode(y2,top=3,layer=1,pred=pred_cata)
	print("\t\t" + ret_msg[0] + ", ", ret_prob[0])
	#for i in range(len(ret_msg)):
	#	print(str(i)+": "+ret_msg[i],ret_prob[i])
	# print(y)
