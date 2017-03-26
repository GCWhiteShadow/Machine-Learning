import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv('train.csv')

df = df.drop(['Name','Ticket','Cabin'], axis = 1)

age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(age_mean)
df['Embarked'] = df['Embarked'].fillna('S')

df['Gender'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
df = df.drop(['Sex', 'Embarked'], axis=1)

cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]
train_data = df.values

x = tf.placeholder(tf.float32, [None, 9], name = 'x_in') 
y = tf.placeholder(tf.float32, [None, 1], name = 'y_in')

W = tf.Variable(tf.zeros([9, 1]), name = 'Weights')
b = tf.Variable(tf.zeros([1,1]), name = 'biases')
Wx_plus_b = tf.add(tf.matmul(x, W, name = 'matrx_multi'), b, name = 'add_biases')

predict = tf.floor(tf.sigmoid(Wx_plus_b, name = 'Sig') + 0.5, name = 'classi')

entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = Wx_plus_b, labels = y, name = 'sig_cros_entropy')

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(entropy,reduction_indices=[1]), name = 'get_loss')
	tf.summary.scalar('loss', loss)

train_step = tf.train.GradientDescentOptimizer(0.005, name = 'grad_dscnt').minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter( "logs/", sess.graph)

sess.run(init)
for i in range(50000):
	sess.run(train_step,feed_dict={x:train_data[:,2:],y:train_data[:,0:1]})
	if i % 5000 == 0:
		result = sess.run(merged, feed_dict={x:train_data[:,2:],y:train_data[:,0:1]})
		writer.add_summary(result, i)
		print(i)

df_test = pd.read_csv('test.csv')

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test['Age'] = df_test['Age'].fillna(age_mean)

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')],axis=1)
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
test_data = df_test.values

output = sess.run(predict,feed_dict={x:test_data[:,1:]})
output = np.squeeze(output)

result = np.c_[test_data[:,0].astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
df_result[df_result < 0] = 0
df_result.to_csv('titanic_ans.csv', index=False)
