import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import numpy as np

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
	print('Hello')
	return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

def read_data(filename, nclass=20, normalize=True):

    if filename is None:
        raise TypeError

    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", skiprows=1)
    feature_size = data.shape[1]-2  # First col is id, last is class label
    x = data[:, 1:feature_size+1]
    if normalize:
        print('normalize')
        x = x/255
        with open("temp_output255.csv", "wb") as f:
            np.savetxt(f, x.astype(int), fmt="%.2f", delimiter=",")
        #x = (x-(range/2))/range
    y = data[:, -1]
    y_data = np.zeros((len(y), nclass))
    y_data[np.arange(len(y)), y.astype(int)]=1
    x_data = np.reshape(x, [-1, 64, 64, 3])
    #print(x_train_tensor.shape)
    return x_data, y_data


"""
with tf.Session() as sess:
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        new_saver = tf.train.import_meta_graph('./toy_code/model.meta')
        new_saver.restore(sess, "./toy_code/model")
        y_val = sess.run('z:0', feed_dict={'x:0':[-4, 2]})
        print(y_val)
        z = g.get_tensor_by_name('z:0')
        x = g.get_tensor_by_name('x:0')
        grad = tf.gradients(z, x)
        print(sess.run(grad, feed_dict={'x:0': [10,2]}))
""" 
with tf.Session() as sess:
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        new_saver = tf.train.import_meta_graph('./GB/model.meta')
        new_saver.restore(sess, './GB/model')
        n_classes=20
        x_train, y_train = read_data('train.csv', nclass=20)
        val_X, val_y = read_data('valid.csv')
        weights, biases = {},{}
        weights['wc1'] = g.get_tensor_by_name('W0:0')
        weights['wc2'] = g.get_tensor_by_name('W1:0')
        weights['wc3'] = g.get_tensor_by_name('W2:0')
        weights['wc4'] = g.get_tensor_by_name('W3:0')
        weights['wc5'] = g.get_tensor_by_name('W4:0')
        weights['wc6'] = g.get_tensor_by_name('W5:0')
        weights['wd1'] = g.get_tensor_by_name('W6:0')
        weights['out'] = g.get_tensor_by_name('W8:0')

        biases['bc1'] = g.get_tensor_by_name('B0:0')
        biases['bc2'] = g.get_tensor_by_name('B1:0')
        biases['bc3'] = g.get_tensor_by_name('B2:0')
        biases['bc4'] = g.get_tensor_by_name('B3:0')
        biases['bc5'] = g.get_tensor_by_name('B4:0')
        biases['bc6'] = g.get_tensor_by_name('B5:0')
        biases['bd1'] = g.get_tensor_by_name('B6:0')
        biases['out'] = g.get_tensor_by_name('B8:0')
    	
    	#print(weights['wc1'].shape)
        
        #get_grad = tf.gradients(conv6_output[:,0,0,0], x)
        accuracy = g.get_tensor_by_name('accuracy:0')
        is_train = g.get_tensor_by_name('is_train:0')
        keep_prob = g.get_tensor_by_name('keep_prob:0')
        correct_prediction = g.get_tensor_by_name('correct_prediction:0')
        x = g.get_tensor_by_name('x:0')
        y = g.get_tensor_by_name('y:0')
        conv6_output = g.get_tensor_by_name('conv6_output:0')
        acc = sess.run(accuracy, feed_dict={x: val_X, y:val_y, is_train:False, keep_prob:1.0})
        correct_prediction = sess.run(correct_prediction, feed_dict={x: val_X[0:4], y:val_y[0:4], is_train:False, keep_prob:1.0})
        conv6 = sess.run(conv6_output, feed_dict={x: val_X, y:val_y, is_train:False, keep_prob:1.0})
        #grad_check = sess.run(get_grad, feed_dict={x:x_train[0:4],keep_prob: 1.0})
        print('Val accuracy')
        print(acc)
        #print(grad_check[0][0][0][0])
        print('oredction checl')
        print(correct_prediction)

        print('output conv6')
        print(conv6.shape)

        grad = tf.gradients(conv6_output, x)
        grad_op = sess.run(grad, feed_dict={x: x_train[0:10], is_train:False, keep_prob:1.0})
        #print(grad_op[0].shape)
        #print(len(grad_op))
        np.savetxt('grad_original_recd.txt', np.reshape(grad_op[0], [10*64*64*3,]))
        #print(grad_check[0].shape)
        #np.savetxt('weights_wc1_recd.txt', np.reshape(weights['wc1'], [5*5*3*32,]))
        #np.savetxt('grad_im1.txt', np.reshape(grad_check[0][0], [64*64*3,]), fmt='%.5f', delimiter='\n')
        #np.savetxt('grad_im2.txt', np.reshape(grad_check[0][1], [64*64*3,]), fmt='%.5f', delimiter='\n')
        #np.savetxt('grad_im3.txt', np.reshape(grad_check[0][2], [64*64*3,]), fmt='%.5f', delimiter='\n')
        #np.savetxt('grad_im4.txt', np.reshape(grad_check[0][3], [64*64*3,]), fmt='%.5f', delimiter='\n')
        #print(grad_check[0][3])
        #plt.imshow(grad_check[0][3])
        #plt.show()