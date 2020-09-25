from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import skimage as sk
from numpy import random as random
from skimage.transform import rotate
import argparse as ap
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

init_type = -1
n_classes=20
n_value = 10 # for n-pixel attack in Fooling
spn = 50 # number of pixels for adding salt and pepper noise

def conv2d(x, W, b, strides=1, padding_value=1):
    # Conv2D wrapper, with bias and relu activation
    if padding_value == 1:
        x = tf.pad(x, [[0, 0], [padding_value, padding_value], [padding_value, padding_value], [0, 0]], "CONSTANT")
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, padding_value=0): # todo : layer 6 different?
    if padding_value == 1:
        x = tf.pad(x, [[0,0], [padding_value, padding_value], [padding_value, padding_value], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')

def conv_net(x, weights, biases):

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_train)
    
    conv2 = conv2d(bn_conv1, weights['wc2'], biases['bc2'])
    pool1 = maxpool2d(conv2, k=2)
    bn_conv2 = tf.layers.batch_normalization(pool1, training=is_train)

    conv3 = conv2d(bn_conv2, weights['wc3'], biases['bc3'])
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_train)

    conv4 = conv2d(bn_conv3, weights['wc4'], biases['bc4'])
    pool2 = maxpool2d(conv4, k=2)
    bn_conv4 = tf.layers.batch_normalization(pool2, training=is_train)

    conv5 = conv2d(bn_conv4, weights['wc5'], biases['bc5'])
    bn_conv5 = tf.layers.batch_normalization(conv5, training=is_train)

    conv6 = conv2d(bn_conv5, weights['wc6'], biases['bc6'], 1, 0)
    pool3 = maxpool2d(conv6, k=2, padding_value=1)
    bn_conv6 = tf.layers.batch_normalization(pool3, training=is_train)

    conv6_flat = tf.reshape(bn_conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
    drop1 = tf.nn.dropout(conv6_flat, keep_prob)
    fc1 = tf.add(tf.matmul(drop1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    drop2 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.add(tf.matmul(drop2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    drop3 = tf.nn.dropout(fc2, keep_prob)
    out = tf.add(tf.matmul(drop3, weights['out']), biases['out'])
    return out


def init_weights():
    print(init_type)
    tf.set_random_seed(1)
    if init_type == 1:
        weights = {
        'wc1': tf.get_variable('W0', shape=(5,5,3,32), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wc2': tf.get_variable('W1', shape=(5,5,32,32), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wc3': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wc4': tf.get_variable('W3', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wc5': tf.get_variable('W4', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wc6': tf.get_variable('W5', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wd1': tf.get_variable('W6', shape=(6272,256), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        #'wd2': tf.get_variable('W7', shape=(256,1024), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wd2': tf.get_variable('W7', shape=(256,256), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'out': tf.get_variable('W8', shape=(256,n_classes), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        }
        biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bc4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bc5': tf.get_variable('B4', shape=(64), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bc6': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bd1': tf.get_variable('B6', shape=(256), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'bd2': tf.get_variable('B7', shape=(256), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        #'bd2': tf.get_variable('B7', shape=(1024), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'out': tf.get_variable('B8', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        }
    else:
        weights = {
        'wc1': tf.get_variable('W0', shape=(5,5,3,32), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'wc2': tf.get_variable('W1', shape=(5,5,32,32), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'wc3': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'wc4': tf.get_variable('W3', shape=(3,3,64,64), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'wc5': tf.get_variable('W4', shape=(3,3,64,64), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'wc6': tf.get_variable('W5', shape=(3,3,64,128), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'wd1': tf.get_variable('W6', shape=(6272,256), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        #'wd2': tf.get_variable('W7', shape=(256,1024), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'wd2': tf.get_variable('W7', shape=(256,256), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'out': tf.get_variable('W8', shape=(256,n_classes), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        }
        biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bc2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bc4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bc5': tf.get_variable('B4', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bc6': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bd1': tf.get_variable('B6', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        'bd2': tf.get_variable('B7', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        #'bd2': tf.get_variable('B7', shape=(1024), initializer=tf.contrib.layers.xavier_initializer(seed=0)),
        'out': tf.get_variable('B8', shape=(n_classes), initializer=tf.contrib.layers.variance_scaling_initializer(seed=0)),
        }

    return weights, biases

def n_pixel_attack(val_x,n):
    min = 0
    max = 63
    s = val_x.shape[0]
    for img in range(s):
        r = random.randint(min, max, n)
        c = random.randint(min, max, n)
        for i in range(n):
            noise_elem = np.random.randn(3)
            pixel_row = r[i]
            pixel_column = c[i]
            val_x[img][pixel_row][pixel_column][:] += noise_elem
    return val_x

def salt_pepper_noise(val_x,n):
    min = 0
    max = 63
    s = val_x.shape[0]
    for img in range(s):
        r = random.randint(min, max, n)
        c = random.randint(min, max, n)
        num_salt = num_pepper = int(n/2)
        plt.imshow(val_x[img])
        plt.show()
        for i in range(num_salt):
            pixel_row = r[i]
            pixel_column = c[i]
            val_x[img][pixel_row][pixel_column][:] = 1.0

        for i in range(num_pepper,n,1):
            pixel_row = r[i]
            pixel_column = c[i]
            val_x[img][pixel_row][pixel_column][:] = 0.0
        # plt.imshow(val_x[img])
        # plt.show()
    return val_x

"""
Returns an ndarray of input and one-hot vector of y
"""
def read_data(filename, nclass=20, normalize=True):

    if filename is None:
        raise TypeError

    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-2  # First col is id, last is class label
    x = data[:, 1:feature_size+1]
    if normalize:
    	# print('normalize')
    	x = x/255
    	#with open("temp_output255.csv", "wb") as f:
        #	np.savetxt(f, x.astype(int), fmt="%.2f", delimiter=",")
    	#x = (x-(range/2))/range
    y = data[:, -1]
    y_data = np.zeros((len(y), nclass))
    y_data[np.arange(len(y)), y.astype(int)]=1
    x_data = np.reshape(x, [-1, 64, 64, 3])
    #print(x_train_tensor.shape)
    return x_data, y_data

def data_augment(dataAugment, data, label, rotright=1, rotleft=1, rotud=1, blur=1, sharpen=1, m=64, n=64, d=3):
    num = data.shape[0]
    left_right_data = np.zeros_like(data)
    for i in range(num):
        mirror_img = np.fliplr(data[i])
        left_right_data[i] = mirror_img

    upside_down_data = np.zeros_like(data)
    for i in range(num):
        ud_img = np.flipud(data[i])
        upside_down_data[i] = ud_img

    rot90_data = np.zeros_like(data)
    for i in range(num):
        r_img = np.rot90(data[i])
        rot90_data[i] = r_img

    spnoisy_data = salt_pepper_noise(data,spn)

    data = np.append(data, left_right_data, axis=0)
    data = np.append(data, upside_down_data, axis=0)
    data = np.append(data, rot90_data, axis=0)
    data = np.append(data, spnoisy_data, axis=0)

    temp1 = np.append(label,label,axis=0)
    temp2 = np.append(temp1, label,axis=0)
    temp3 = np.append(temp2, label,axis=0)
    label = np.append(temp3, label,axis=0)
    return data, label

def convert_data(x, y, nclass=20):
    y_data = np.zeros((len(y), nclass))
    y_data[np.arange(len(y)), y.astype(int)]=1
    x_data = np.reshape(x, [-1, 64, 64, 3])
    #print(x_train_tensor.shape)
    return x_data, y_data

def read_plain_data(filename, nclass=20):
    if filename is None:
        raise TypeError
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-2  # First col is id, last is class label
    x = data[:, 1:feature_size+1]
    y = data[:, -1]
    return x,y

def fool(val_x):
    n = val_x.shape[0]
    for i in range(n):
        noise = 0.001*val_x[i]
        val_x[i] = 0.3*val_x[i] + noise
    return val_x

def write_loss_to_file(x, filename):
    if filename is None:
        raise TypeError
    np.savetxt(filename, x, fmt='%.4f')


parser = ap.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--batch_size")
parser.add_argument("--init")
parser.add_argument("--save_dir")
parser.add_argument("--epochs")
parser.add_argument("--dataAugment")
parser.add_argument("--train")
parser.add_argument("--val")
parser.add_argument("--test")
args = parser.parse_args()

# training_iters = 120
# learning_rate = 0.001
# batch_size = 128
# n_classes = 20
# keep_prob_train = 0.4
# keep_prob_one = 1.0 # For test time

learning_rate = float(args.lr)
batch_size = int(args.batch_size)
init_type = int(args.init)
save_dir = str(args.save_dir)
training_iters = int(args.epochs)
dataAugment = int(args.dataAugment)
train_file = str(args.train)
val_file = str(args.val)
test_file = str(args.test)

n_classes = 20
keep_prob_train = 0.4
keep_prob_one = 1.0 # For test time


x_train, y_train = read_data(train_file, nclass=20)

if dataAugment == 1:
    x_train, y_train = data_augment(1, x_train, y_train)

val_X, val_y = read_data(val_file)
#val_X = fool(val_X)
#val_X = n_pixel_attack(val_X,n_value)
x = tf.placeholder("float", [None, 64,64,3])
y = tf.placeholder("float", [None, n_classes])
is_train = tf.placeholder(tf.bool, name="is_train")
keep_prob = tf.placeholder(tf.float32)
weights, biases = init_weights()
pred = conv_net(x, weights, biases)

x_norm_test = tf.layers.batch_normalization(pred, training=is_train)
prediction_vec = tf.nn.softmax(x_norm_test)
pred_label_test = tf.argmax(prediction_vec,1)

x_norm = tf.layers.batch_normalization(pred, training=is_train)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x_norm, labels=y))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()

#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    print('train iters:' + str(training_iters))
    for i in range(training_iters):
        for batch in range(len(x_train)//batch_size):
            batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
            batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
            opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, is_train: True, keep_prob:keep_prob_train})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y , is_train: False, keep_prob:keep_prob_one})
        print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        print("Optimization Finished!")
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: val_X, y:val_y, is_train:False, keep_prob:keep_prob_one})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Validation Accuracy:","{:.5f}".format(test_acc))

    write_loss_to_file(train_loss, 'train_loss.txt')
    write_loss_to_file(test_loss, 'val_loss.txt')
    # write_loss_to_file(train_accuracy, 'train_accuracy.txt')
    # write_loss_to_file(test_acc. 'val_accuracy.txt')

   	#For Test data
    filename = test_file
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-1  # First col is id
    id1 = data[:, 0]
    x_test = data[:, 1:feature_size+1]
    x_test = x_test/255
    x_test = np.reshape(x_test, [-1, 64, 64, 3])
    predictions = sess.run([pred_label_test], feed_dict={x:x_test, is_train:False, keep_prob:1.0})
    y_pred = np.array(predictions)
    y_pred = np.reshape(y_pred, [y_pred.shape[1],])
    final_output = np.concatenate((id1[:, None], y_pred[:, None]), axis=1)
    with open("test_submission.csv", "wb") as f:
    	f.write(b'id,label\n')
    	np.savetxt(f, final_output.astype(int), fmt='%i', delimiter=",")
    summary_writer.close()
