import numpy as np
import os
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
from keras.models import load_model
import dg as dg_all
import datagen as dg
from tqdm import tqdm

all_size = 133235
test_size = 2000
batch_size = 100
SUPER_CLASSES = 4

LABEL = ["corn","grape","tomato","apple"]
ADD_LABEL = [0,4,8,18]

label_file = open("tree/label.txt")
label_name = []
while 1:
	line = label_file.readline()
	if not line:
		break
	else:
		label_name.append(line)
label_file.close()

def get_label_name(cl):
	return label_name[cl].split('\n')[0]

def decode(y,batch_size):	#get the name of label and the probability
	rank = np.argsort(-y)
	#print(rank)
	ret_pred = rank[:,0]
	return ret_pred



def batch_pred(pred,batch_data):	#predict a batch of data
	ret_pred = np.zeros(batch_size)
	temp_list = np.zeros([SUPER_CLASSES,batch_size])	#set a list for every class
	class_cnt = np.zeros(SUPER_CLASSES)
	
	class_cnt = class_cnt.astype(np.int32)
	temp_list = temp_list.astype(np.int32)
	ret_pred = ret_pred.astype(np.int32)
	
	for i,p in enumerate(pred):
		temp_list[p][class_cnt[p]] = i	# temp_list to record the index of class p in pred
		class_cnt[p] += 1	# class_cnt to record the total number of class p in pred
		
	for cl in tqdm(range(SUPER_CLASSES)):
		model = load_model("tree/model/"+LABEL[cl]+"_413.h5")
		if class_cnt[cl] > 0:
			test_arr = batch_data[temp_list[cl]][0]
			for c in range(1,class_cnt[cl]):
				test_arr = np.concatenate((test_arr,batch_data[temp_list[cl]][c]))
			test_arr.resize((class_cnt[cl],224,224,3))
			y = model.predict(test_arr,class_cnt[cl])
			y = np.array(y)
			pred = decode(y,batch_size) + ADD_LABEL[cl]
		for c in range(class_cnt[cl]):
			ret_pred[temp_list[cl][c]] = pred[c]
		
	"""
	for CL in tqdm(range(SUPER_CLASSES)):	#4
		model = load_model("tree/model/"+LABEL[CL]+"_413.h5")
		for c in range(class_cnt[CL]):	#class cnt
			test_data = batch_data[temp_list[CL][c]]
			test_data = test_data[np.newaxis,:]
			y = model.predict(test_data,1)
			pred = decode(y,batch_size) + ADD_LABEL[CL]
			ret_pred[temp_list[CL][c]] = pred[0]
	"""
	
	
	return ret_pred



if __name__ == '__main__':
	filelist = dg_all.file_gen("dataset1")
	model = load_model("tree/model/all_415.h5")

	times = test_size // batch_size
	test_data = []
	test_label = []
	
	Acc_in_all = 0.0	
	P_in_all = 0.0	
	R_in_all = 0.0	
	F_in_all = 0.0
	
	for t in tqdm(range(times)):
		output_msg = ''
		output_msg += "==============================\r\n"
		batch_data, batch_label, _, _, filelist = dg_all.image_generator(filelist,batch_size,1,method=1)
		


		y = model.predict(batch_data,batch_size)
		pred = decode(y,batch_size)
		real = np.argmax(batch_label,axis=1)
		#print(pred)
		
		
		sub_pred = batch_pred(pred,batch_data)
		#print(sub_pred)
		#print(real)
		
		
		
		class_cnt = 0
		class_num = np.zeros(22) - 1	#number of this class in real label
		class_right = np.zeros(22)
		class_wrong = np.zeros(22)	#wrong items been classified to this class
		right_cnt = 0
		for item in range(batch_size):
			pr = sub_pred[item]
			rl = real[item]
			if class_num[rl] == -1:
				class_num[rl] += 1
				class_cnt += 1
			class_num[rl] += 1
			if pr == rl:	#right
				right_cnt += 1
				class_right[pr] += 1
			else:	#wrong
				class_wrong[pr] += 1
			
		#Accuracy
		accuracy = right_cnt / batch_size
		#Precision
		for item in range(22):
			if class_right[item] + class_wrong[item] == 0:	#no this class
				class_wrong[item] -= 1	#in case divided by zero
		precision = np.array(class_right) / np.array(class_right + class_wrong)
		#Recall
		recall = np.array(class_right) / np.array(class_num)
		recall_temp = recall
		#F1 = 2 * (P * R) / (P + R)
		for item in range(22):
			if precision[item] + recall_temp[item] == 0:	#no this class
				recall_temp[item] -= 1	#in case divided by zero
		F = 2 * (precision * recall_temp) / (precision + recall_temp)
		for cl in range(22):
			if recall[cl] < 0:
				recall[cl] = -0
		
		mean_P = sum(precision) / class_cnt
		mean_R = sum(recall) / class_cnt
		mean_F = sum(F) / class_cnt
		
		output_msg += "Batch " + str(t) + "\r\n"
		#output_msg += "Real: " + str(real) + "\r\n"
		#output_msg += "Pred: " + str(sub_pred) + "\r\n"
		output_msg += "Accuracy: " + str(accuracy) + "\r\n"
		output_msg += "Mean-P: " + str(mean_P) + "\r\n"
		output_msg += "Mean-R: " + str(mean_R) + "\r\n"
		output_msg += "Mean-F: " + str(mean_F) + "\r\n"
		
		Acc_in_all += accuracy	
		P_in_all += mean_P
		R_in_all += mean_R
		F_in_all += mean_F
		
		"""
		for cl in range(22):
			output_msg += ("Class " + str(cl) + ": " + get_label_name(cl) + "\r\n"
			+ ": P: " + str(precision[cl]) + " R:" + str(recall[cl])
			+ " F: " + str(F[cl]) + "\r\n")
		"""
	
		f = open("tree/result.txt",'a')
		f.write(output_msg)
		f.close()
		
	Acc_in_all /= t	+1
	P_in_all /= t+1
	R_in_all /= t+1
	F_in_all /= t+1
	
	print("In all: acc =",Acc_in_all,", P =",P_in_all,", R =",R_in_all,", F =",F_in_all)
