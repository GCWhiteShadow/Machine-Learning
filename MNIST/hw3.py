import pandas as pd
import numpy as np
import tensorflow as tf

def compute_accuracy(v_xs, v_ys):
    global predict
    y_pre = sess.run(predict, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = weight_variable([in_size, out_size])
    biases = bias_variable([out_size])
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


df = pd.read_csv('train.csv')
df = pd.concat([pd.get_dummies(df['label'], prefix='label'), df], axis=1)
df = df.drop(['label'], axis=1)

##print(df)
train_data = df.values
train_data = train_data.astype(float)
train_data[:, 10:] = train_data[:, 10:] / 255.
train_batch = np.array_split(train_data, 420)

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1,28,28,1])


W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
predict = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
correct_prediction = tf.argmax(predict, 1)


'''
l1 = add_layer(h_pool2_flat, 7*7*64, 1024, tf.nn.relu)
l1_drop = tf.nn.dropout(l1, keep_prob)
prediction = add_layer(l1_drop, 1024, 10, tf.nn.softmax)
'''
with tf.name_scope('graph'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = ys))
    rec_cross = tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs-cnn/", sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(200):
	batch_xs = train_batch[i%100][:,10:]
	batch_ys = train_batch[i%100][:,:10]
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})						
	if i % 20 == 0:
		rs = sess.run(merged ,feed_dict={xs:batch_xs, ys: batch_ys, keep_prob: 0.5})
		writer.add_summary(rs, i)
		print(compute_accuracy(train_data[:50,10:], train_data[:50,0:10]))
		print(sess.run(correct_prediction, feed_dict={xs: train_data[:20, 10:], keep_prob: 1}))

df = pd.read_csv('test.csv')
test_data = df.values
test_data = test_data.astype(float)
test_data = test_data / 255.

output = sess.run(correct_prediction, feed_dict={xs: test_data[:,:], keep_prob: 1})
ans = np.zeros((test_data.shape[0],2), dtype  = int)

for i in range(output.shape[0]):
    ans[i,0] = i+1
    ans[i,1] = output[i]

#print(result)
df_result = pd.DataFrame(ans, columns=['ImageId', 'Label'])
df_result.to_csv('MNIST_result.csv', index=False)

