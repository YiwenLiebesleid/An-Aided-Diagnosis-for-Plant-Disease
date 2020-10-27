import numpy as np
import os
from PIL import Image
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications import *
from keras.models import load_model
import datagen as dg
from tqdm import tqdm

all_size = 133235
test_size = 5000
batch_size = 100

label_file = open("label.txt")
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


if __name__ == '__main__':
	filelist = dg.file_gen()
	model = load_model("test_413.h5")
	
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
		batch_data, batch_label, _, _, filelist = dg.image_generator(filelist,batch_size,1)
		
		y = model.predict(batch_data,batch_size)
		pred = decode(y,batch_size)
		real = np.argmax(batch_label,axis=1)
		
		class_cnt = 0
		class_num = np.zeros(22) - 1	#number of this class in real label
		class_right = np.zeros(22)
		class_wrong = np.zeros(22)	#wrong items been classified to this class
		right_cnt = 0
		for item in range(batch_size):
			pr = pred[item]
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
		#output_msg += "Pred: " + str(pred) + "\r\n"
		output_msg += "Accuracy: " + str(accuracy) + "\r\n"
		output_msg += "Mean-P: " + str(mean_P) + "\r\n"
		output_msg += "Mean-R: " + str(mean_R) + "\r\n"
		output_msg += "Mean-F: " + str(mean_F) + "\r\n"
		
		Acc_in_all += accuracy	
		P_in_all += mean_P
		R_in_all += mean_R
		F_in_all += mean_F
		
		for cl in range(22):
			output_msg += ("Class " + str(cl) + ": " + get_label_name(cl) + "\r\n"
			+ ": P: " + str(precision[cl]) + " R:" + str(recall[cl])
			+ " F: " + str(F[cl]) + "\r\n")
	
		f = open("resnet/result.txt",'a')
		f.write(output_msg)
		f.close()
		
	Acc_in_all /= t	
	P_in_all /= t
	R_in_all /= t
	F_in_all /= t
	
	print("In all: acc =",Acc_in_all,", P =",P_in_all,", R =",R_in_all,", F =",F_in_all)
