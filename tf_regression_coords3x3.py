"""
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)

optimisation function (optimiser) > minimise cost (AdamOptimizer ... Gradient Descent, AdaGrad, ...)

backpropagation

feed forward + backpropagation = epoch
"""
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('coords3x3_full_copy.dat')
df['A'] = df['A'].apply(str)
df['B'] = df['B'].apply(str)
df['C'] = df['C'].apply(str)
df['D'] = df['D'].apply(str)
df['E'] = df['E'].apply(str)
df['F'] = df['F'].apply(str)
df['G'] = df['G'].apply(str)
df['H'] = df['H'].apply(str)
df['I'] = df['I'].apply(str)
df['J'] = df['J'].apply(str)
bigtest_set = df.sample(frac=0.01, replace=True)
df = df.sample(frac=0.05, replace=True)
print(len(df))

numeric_cols = ['nbins', 'dir_real', 'dir_artif']

x_num = df[numeric_cols].as_matrix()
# print(len(x_num))
x_num_bigtest = bigtest_set[numeric_cols].as_matrix()

max_x = np.amax(x_num_bigtest, 0)
# print(max_x)
x_num = x_num / max_x
x_num_bigtest = x_num_bigtest / max_x

cat = df.drop(numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)

# print(len(cat))
cat_bigtest = bigtest_set.drop(numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)

x_cat = cat.to_dict(orient='records')
x_cat_bigtest = cat_bigtest.to_dict(orient='records')

# vec = DictVectorizer(sparse=False)
# vec = DictVectorizer(sparse=False)
# vec = vec.fit(x_cat_bigtest)
#
# with open("pickle_vectorizer.pickle", 'wb') as out:
#     pickle.dump(vec, out)

vec = pickle.load(open('pickle_vectorizer.pickle', 'rb'))

vec_x_cat_bigtest = vec.transform(x_cat_bigtest)
vec_x_cat = vec.transform(x_cat)
# print(len(vec_x_cat))

XX = np.hstack((x_num, vec_x_cat))
# print(len(X))
X_bigtest = np.hstack((x_num_bigtest, vec_x_cat_bigtest))

yy = df[['lcoe']].as_matrix()
y_bigtest = bigtest_set[['lcoe']].as_matrix()

n_nodes_hl1 = 42
n_nodes_hl2 = 19
n_nodes_hl3 = 4
n_classes = 1

batch_size = 200

x_train, x_test, y_train, y_test = train_test_split(XX, yy, test_size=0.2, random_state=2)

# matrix: height x weight
x = tf.placeholder(tf.float32, [None, 42])
y = tf.placeholder(tf.float32)


def neural_network_model(data):

    # (input_data * weights) + biases

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([42, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # tf.add is sum, tf.matmul is matrix multiplication.
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.square(prediction - y))
    # cost = tf.reduce_sum(tf.square(prediction - y)) / 4.0
    # cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(prediction, y))))
    #                                            learning_rate = 0.001
    # optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    optimiser = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + back propagation
    hm_epochs = 2000

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        epoch_loss = 0

        for epoch in range(hm_epochs):
            epoch_loss_old = epoch_loss
            epoch_loss = 0
            total_batch = int(len(x_train) / batch_size)
            # Loop over all batches
            for i in range(total_batch - 1):
                x_epoch = x_train[i * batch_size:(i + 1) * batch_size]
                y_epoch = y_train[i * batch_size:(i + 1) * batch_size]
                _, c = sess.run([optimiser, cost], feed_dict={x: x_epoch, y: y_epoch})
                epoch_loss += c

            losses.append(epoch_loss)
            print('Epoch', epoch, 'completed out of', hm_epochs, 'Loss:', epoch_loss)
            if math.fabs(epoch_loss_old - epoch_loss) <= 0.1:
                if epoch > 0:
                    break
                    # pass

        pred_y = sess.run(prediction, feed_dict={x: x_test})
        # print(pred_y.dtype)
        # accuracy = tf.metrics.accuracy(bigtest_set[['lcoe']], pred_y)
        # print("MSE: %.4f" % sess.run(accuracy))
        mse = tf.reduce_sum(tf.square(pred_y - y_test))
        squares = tf.reduce_sum(tf.square(tf.subtract(y_test, np.average(y_test))))
        r2_score = tf.subtract(1.0, tf.cast(tf.divide(mse, squares), 'float32'))
        accuracy = sess.run(r2_score, feed_dict={x: x_test, y: y_test})
        print("R^2 score: %.4f" % accuracy)

        plt.figure(1)
        ax = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax.scatter(y_test, pred_y, s=1)
        ax2.scatter(range(len(losses[15:])), losses[15:])
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        plt.show()


train_neural_network(x)
