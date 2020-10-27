import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from keras.utils.np_utils import to_categorical

CLASS_NUM = 4

# input path -> image
def get_input(path):
    image = Image.open(path)
#    return(imread(path))
    return(np.array(image))
# input path -> label
def get_label(path):
	temp_label = np.int64(path.split('/')[-1].split('-')[0])
	if temp_label >= 18 and temp_label <= 21:	#apple
		temp_label = 3
	elif temp_label >= 0 and temp_label <= 3:	#corn
		temp_label = 0
	elif temp_label >= 4 and temp_label <= 7:	#grape
		temp_label = 1
	elif temp_label >= 8 and temp_label <= 17:	#tomato
		temp_label = 2
	else:
		temp_label = temp_label
		#do nothing
	return temp_label

def image_generator(files,size,scale,classnum=4):	#scale is a (0,1)
    batch_paths = np.random.choice(a=files,size=size,replace=False)	#choose a batchsize of files (path)
    train_num = int(size * scale)
    CLASS_NUM = classnum

    batch_data = []
    batch_label = []
    rest_data = []
    rest_label = []

    for input_path in batch_paths[0:train_num]:
        data = get_input(input_path)
        label = get_label(input_path)

        # data = preprocess_input(image=input)
        batch_data += [data]
        batch_label += [label]

        files.remove(input_path)

    for input_path in batch_paths[train_num:size]:
        data = get_input(input_path)
        label = get_label(input_path)

        # data = preprocess_input(image=input)
        rest_data += [data]
        rest_label += [label]

        files.remove(input_path)
    
    batch_data = np.array(batch_data)
    batch_label = to_categorical(np.array(batch_label),num_classes=CLASS_NUM)
    rest_data = np.array(rest_data)
    rest_label = to_categorical(np.array(rest_label),num_classes=CLASS_NUM)

    return batch_data, batch_label, rest_data, rest_label, files



def file_gen(path):
    dirs = os.listdir(path)
    print(dirs)
    filelist = []
    for dir in tqdm(dirs):
        subpath = os.path.join(os.path.join(path,dir),"cut")
        files = os.listdir(subpath)
        for f in files:
            filelist.append(os.path.join(subpath,f))
    return filelist
