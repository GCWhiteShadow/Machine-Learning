import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('test.csv')
test_data = df.values
test_data = test_data.astype(float)
test_data = test_data / 255.

#print(test_data.shape)
df = pd.read_csv('train.csv')
df = pd.concat([pd.get_dummies(df['label'], prefix='label'), df], axis=1)
df = df.drop(['label'], axis=1)

##print(df)
train_data = df.values
train_data = train_data.astype(float)
train_data[:, 10:] = train_data[:, 10:] / 255.

#print(train_data.shape)
train_batch = np.array_split(train_data, 420)

learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1
examples_to_show = 20

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)

def compute_accuracy(v_xs, v_ys):
	global predict
	y_pre = sess.run(predict, feed_dict={X: v_xs, keep_prob: 1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={X: v_xs, ys: v_ys, keep_prob: 1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

	# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 32

weights = {
	'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
	'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
	'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}

e_h1 = tf.summary.histogram('encoder_h1', weights['encoder_h1'])
e_b1 = tf.summary.histogram('encoder_b1', biases['encoder_b1'])
e_h2 = tf.summary.histogram('encoder_h2', weights['encoder_h2'])
e_b2 = tf.summary.histogram('encoder_b2', biases['encoder_b2'])
e_h3 = tf.summary.histogram('encoder_h3', weights['encoder_h3'])
e_b3 = tf.summary.histogram('encoder_b3', biases['encoder_b3'])
d_h1 = tf.summary.histogram('decoder_h3', weights['decoder_h3'])
d_b1 = tf.summary.histogram('decoder_b3', biases['decoder_b3'])
d_h2 = tf.summary.histogram('decoder_h3', weights['decoder_h3'])
d_b2 = tf.summary.histogram('decoder_b3', biases['decoder_b3'])
d_h3 = tf.summary.histogram('decoder_h3', weights['decoder_h3'])
d_b3 = tf.summary.histogram('decoder_b3', biases['decoder_b3'])
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),biases['encoder_b3']))
	return layer_3

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
	layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),biases['decoder_b3']))
	return layer_3

# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, [None, n_input])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
 #Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error

W_fc1 = weight_variable([n_input, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(X ,W_fc1)+b_fc1)
drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

predict = tf.matmul(drop,W_fc2)+b_fc2

with tf.name_scope('graph'):
	cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
	rec_cost = tf.summary.scalar('cost', cost)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels =  ys))
	rec_cross = tf.summary.scalar('cross_entropy', cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

correct_prediction = tf.argmax(predict, 1)
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
	merged = tf.summary.merge([e_h1,e_b1,e_h2,e_b2,e_h3,e_b3,d_h1,d_b1,d_h2,d_b2,d_h3,d_b3,rec_cost])
	#merged_2 = tf.summary.merge([rec_cross, rec_acc])
	writer = tf.summary.FileWriter("logs-FC/", sess.graph)
	sess.run(init)
	total_batch = int(42000/batch_size)
	# Training cycle
	'''
	for epoch in range(training_epochs):
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = train_batch[i%100][:,10:]
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
		# Display logs per epoch step
		
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
			#print(sess.run(y_true, feed_dict={X: batch_xs}))
			#print(sess.run(y_pred, feed_dict={X: batch_xs}))
		rs = sess.run(merged ,feed_dict={X:batch_xs})
		writer.add_summary(rs, epoch)
	print("Optimization Finished!")
	# # Applying encode and decode over test set
	encode_decode = sess.run(y_pred, feed_dict={X: test_data[:examples_to_show]})
	# Compare original images with their reconstructions
	f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
	for i in range(examples_to_show):
		a[0][i].imshow(np.reshape(test_data[i], (28, 28)))
		a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
	plt.show()
	'''
	for i in range(5000):
		batch_xs = train_batch[i%420][:,10:]
		batch_ys = train_batch[i%420][:,:10]
		#print(sess.run(X[0], feed_dict={X: train_data[:, 10:]}))
		sess.run(train_step, feed_dict={X: batch_xs, ys: batch_ys, keep_prob: 0.5})
		if i % 500 == 0:
			print(compute_accuracy(train_data[:100, 10:],train_data[:100, :10]))
			print(sess.run(correct_prediction, feed_dict={X: train_data[:20, 10:], keep_prob: 1}))
		if i % 100 == 0:
			rs = sess.run(tf.summary.merge([rec_cross]) ,feed_dict={X: batch_xs, ys: batch_ys, keep_prob: 0.5})
			writer.add_summary(rs, i)

	output = sess.run(correct_prediction, feed_dict={X: test_data[:,:], keep_prob: 1})
	print(output[:20])

ans = np.zeros((test_data.shape[0],2),dtype  = int)

for i in range(output.shape[0]):
	ans[i,0] = i+1
	ans[i,1] = output[i]

#print(result)
df_result = pd.DataFrame(ans, columns=['ImageId', 'Label'])
df_result.to_csv('MNIST_result.csv', index=False)
