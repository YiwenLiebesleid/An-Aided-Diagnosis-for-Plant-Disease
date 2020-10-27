import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import os
from keras.utils import np_utils
from tqdm import tqdm
import numpy as np

BATCHSIZE = 10

files = os.listdir('G:\Thesis\dataset\Tomato')
trainpaths = []
for file in files:
    trainpaths.append("G:\Thesis\dataset\Tomato\\"+file+"\\cut")

batches = []
for cate_id,trainpath in enumerate(trainpaths):
    listdir = os.listdir(trainpath)
    batches.append([])
    print(trainpath)
    for f in tqdm(listdir[00:900]):
        batches[cate_id].append(utils.load_image(os.path.join(trainpath,f)))

labels = []
for cate_id,cate_batches in enumerate(batches):
    length = len(cate_batches)
    for i in range(length):
        labels.append(cate_id)

new_batches = []
for batch in batches:
    for img in batch:
        new_batches.append(img)
batches = new_batches

with tf.device('/cpu:0'):
    length = len(labels)
    shuffle_idx = np.random.randint(0,100000000,length)
    shuffle_idx =  np.argsort(shuffle_idx)
    for i in range(length):
        batches.append(batches[shuffle_idx[i]])
        labels.append(labels[shuffle_idx[i]])
    batches = batches[length:]
    labels = labels[length:]

    train_batches = batches[0 : int(length*0.8)]
    valid_batches = batches[int(length*0.8):]
    train_labels = labels[0 : int(length*0.8)]
    valid_labels = labels[int(length*0.8):]

    sess = tf.Session()

    images = tf.placeholder(tf.float32, [BATCHSIZE, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [BATCHSIZE,10])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./test-save-tomato.npy',dropout=1)
    vgg.build(images, train_mode)
    cost = -tf.reduce_sum(true_out * tf.log(vgg.prob))
    cost = tf.reduce_mean(cost) # cross entropy
    optimizer = tf.train.AdamOptimizer(0.00001)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train = optimizer.apply_gradients(capped_gvs)

    sess.run(tf.global_variables_initializer())


    batch_num = len(train_batches)//BATCHSIZE
    train_loss = 0
    train_accuracy = 0
    for batch_id in tqdm(range(batch_num)):
        batch = train_batches[batch_id*BATCHSIZE:(batch_id + 1) * BATCHSIZE]
        batch = np.array(batch)
        label = train_labels[batch_id*BATCHSIZE:(batch_id + 1) * BATCHSIZE]
        label = np.array(label)
        label = label.reshape((-1))
        label = np_utils.to_categorical(label, num_classes=10)

        cost_value,_ , prob_value= sess.run([cost,train,vgg.prob], feed_dict={images: batch, true_out: label, train_mode: True})
        train_loss += cost_value
        accuracy = np.sum(np.argmax(label,axis = 1) == np.argmax(prob_value,axis = 1))/BATCHSIZE
        train_accuracy += accuracy
        print('batch = {}, cost = {} avg_loss = {} accu = {} avg_accu = {}'.format(
            batch_id,cost_value,train_loss/(batch_id + 1),accuracy,train_accuracy/(batch_id + 1)))

    # test save
    vgg.save_npy(sess, './test-save-tomato.npy')

    batch_num_test = len(valid_batches)//BATCHSIZE
    test_accuracy = 0
    for batch_id in range(batch_num_test):
        batch = valid_batches[batch_id*BATCHSIZE:(batch_id+1)*BATCHSIZE]
        batch = np.array(batch)
        label = valid_labels[batch_id * BATCHSIZE:(batch_id+1)*BATCHSIZE]
        label = np.array(label)
        label = label.reshape((-1))
        label = np_utils.to_categorical(label, num_classes=10)

        prob_value = sess.run(vgg.prob,feed_dict={images: batch, true_out: label, train_mode: False})
        accuracy = np.sum(np.argmax(label, axis=1) == np.argmax(prob_value, axis=1)) / BATCHSIZE
        test_accuracy += accuracy
        print('batch = {} accu = {} avg_accu = {}'.format(
            batch_id, accuracy, test_accuracy / (batch_id + 1)))
