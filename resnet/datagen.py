import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from keras.utils.np_utils import to_categorical

# input path -> image
def get_input(path):
    image = Image.open(path)
#    return(imread(path))
    return(np.array(image))

# input path -> label
def get_label(path):
    return np.int64(path.split('/')[-1].split('-')[0])

def image_generator(files,size,scale):	#scale is a (0,1)
    batch_paths = np.random.choice(a=files,size=size,replace=False)	#choose a batchsize of files (path)
    train_num = int(size * scale)
    
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
    batch_label = to_categorical(np.array(batch_label),num_classes=22)
    rest_data = np.array(rest_data)
    rest_label = to_categorical(np.array(rest_label),num_classes=22)

    return batch_data, batch_label, rest_data, rest_label, files

def file_gen():
    path = "./dataset1"
    dirs = os.listdir(path)
    filelist = []
    for dir in tqdm(dirs):
        subpath = os.path.join(os.path.join(path,dir),"cut")
        files = os.listdir(subpath)
        for f in files:
            filelist.append(os.path.join(subpath,f))
    return filelist
