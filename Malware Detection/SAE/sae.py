import numpy as np
import tensorflow as tf

###################################################################
#                   Initial Variables Settings                    #
###################################################################
weights, biases = [], []
loss = 
batch_size = 100
learning_rate = 
print_step = 300
epoch = 6000
#input_dim = [1000, 500, 250, 100]
hidden_dims = [500, 250, 100, 50]
d_type = tf.float32
###################################################################

#Purpose:   random get a batch from dataset
#Input:     src - the source dataset
#Output:    the desire batch
def get_batch(src):
    tmp = np.random.choice(len(src), batch_size, replace = False)
    batch = []
    for i in range(batch_size):
        batch.append(src[tmp[i]])
    return batch

#Purpose:   the activation function
#Input:     linear - the data would like to train
#Ouput:     the data after train
def activaion(linear):
    return tf.nn.relu(linear, name = 'encoded')

#Purpose:   set a random (w,b) and start train a single layer and store the (w,b)
#Input:     src - the input data(matrix)
#           hidden_dim - hidden layer dimension
#Output:    the data matrix after training
def run(src, hidden_dim):
    input_dim = len(src[0])
    sess = tf.Session()
    x = tf.placeholder(dtype = d_type, shape = [None, input_dim], name = 'data')
    
    encode = {
        'weights' : tf.Variable(tf.truncated_normal([input_dim, hidden_dim], 
                    dtype = d_type)),
        'biases' : tf.Variable(tf.truncated_normal([hidden_dim], dtype = d_type))
    }
    decode = {
        'weights' : tf.transpose(encode['weights'])
        'biases' : tf.Variable(tf.truncated_normal([input_dim], dtype = d_type))
    }
    encoded = activaion(tf.matmul(x, encode['weights']) + encode['biases'])
    decoded = tf.matmul(encoded, decode['weights']) + decode['biases']

    loss = -tf.reduce_mean(x_ * tf.log(decoded))
    train_op = tf.train.AdamOptimizer(learning_rate). minimize(loss)

    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch = get_batch(src)
        sess.run(train_op, feed_dict = {x: batch})
        if i % print_step == 0 & i != 0:
            l = sess.run(loss, feed_dict = {x: src})
            print('epoch {0}: loss = {1}'.format(i, l))

    weights.append(sess.run(encode['weights']))
    biases.append(sess.run(encode['biases']))
    return sess.run(encoded, feed_dict={x: src})

#Purpose:   to make the previous layers' (w,b) to the best
#Input:     training data(bottleneck input)
#Output:    none
def fine_tuning(src):


#Purpose:   as title, pre training
#Input:     src - training data
#Output:    none
def pre_training(src):
    data = np.copy(src)
    for i in range(len(hidden_dims) - 1):
        data = run(data, hidden_dims[i])
    fine_tuning(data)