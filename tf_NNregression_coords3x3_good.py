from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn import datasets, linear_model
from sklearn import model_selection
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd

df = pd.read_csv('coords3x3_full_copy.dat')
# df['A'] = df['A'].apply(str)
# df['B'] = df['B'].apply(str)
# df['C'] = df['C'].apply(str)
# df['D'] = df['D'].apply(str)
# df['E'] = df['E'].apply(str)
# df['F'] = df['F'].apply(str)
# df['G'] = df['G'].apply(str)
# df['H'] = df['H'].apply(str)
# df['I'] = df['I'].apply(str)
# df['J'] = df['J'].apply(str)
bigtest_set = df.sample(frac=0.01, replace=True)
df = df.sample(frac=0.01, replace=True)

numeric_cols = ['nbins', 'dir_real', 'dir_artif']
#
x_num = df[numeric_cols].as_matrix()
# # print(len(x_num))
x_num_bigtest = bigtest_set[numeric_cols].as_matrix()
#
max_x = np.amax(x_num_bigtest, 0)
# # print(max_x)
x_num = x_num / max_x
x_num_bigtest = x_num_bigtest / max_x
#
cat = df.drop(numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)
#
# # print(len(cat))
cat_bigtest = bigtest_set.drop(numeric_cols + ['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1)
#
x_cat = cat.to_dict(orient='records')
x_cat_bigtest = cat_bigtest.to_dict(orient='records')

vec = DictVectorizer(sparse=False)
vec = vec.fit(x_cat_bigtest)
# #
# # with open("pickle_vectorizer.pickle", 'wb') as out:
# #     pickle.dump(vec, out)
#
# # vec = pickle.load(open('pickle_vectorizer.pickle', 'rb'))
#
vec_x_cat_bigtest = vec.transform(x_cat_bigtest)
vec_x_cat = vec.transform(x_cat)
# print(len(vec_x_cat))
#
x = np.hstack((x_num, vec_x_cat))
# # print(len(X))
X_bigtest = np.hstack((x_num_bigtest, vec_x_cat_bigtest))
#
y = df[['lcoe']].as_matrix()
y_bigtest = bigtest_set[['lcoe']].as_matrix()
# print(len(x), x[:10])
# print(len(y), y[:10])

# x = df.drop(['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1).as_matrix()
# y = df[['lcoe']].as_matrix()
x_big = bigtest_set.drop(['lcoe', 'aep', 'time', 'power_calls', 'ct_calls'], 1).as_matrix()
y_big = bigtest_set[['lcoe']].as_matrix()

y = np.reshape(y, [y.shape[0], 1])
y_big = np.reshape(y_big, [y_big.shape[0], 1])
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42)

total_len = X_train.shape[0]

# Parameters
learning_rate = 0.001
training_epochs = 500
batch_size = 10
display_step = 10
# dropout_rate = 0.9
# Network Parameters
n_hidden_1 = 42  # 1st layer number of features
n_hidden_2 = 310  # 2nd layer number of features
n_hidden_3 = 310
n_hidden_4 = 300
n_input = X_train.shape[1]
n_classes = 1

# tf Graph input
x = tf.placeholder("float", [None, 42])
y = tf.placeholder("float", [None, 1])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


def neuralnetwork():
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1)),
        'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1)),
        'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1)),
        'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    mse = tf.reduce_mean(tf.square(pred - y_big))
    squares = tf.reduce_sum(tf.square(tf.subtract(y_big, tf.reduce_mean(y_big))))
    good = tf.subtract(1.0, tf.divide(mse, tf.cast(squares, tf.float32)))

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training cycle
        losses = []
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x = X_train[i * batch_size:(i + 1) * batch_size]
                batch_y = Y_train[i * batch_size:(i + 1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                                       y: batch_y})
                # Compute average loss
                avg_cost += c# / total_batch
                losses.append(avg_cost)
            # sample prediction
            label_value = batch_y
            estimate = p
            # err = label_value - estimate

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("num batch:", total_batch)
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print("[*]----------------------------")
                for i in range(3):
                    print("label value:", label_value[i], "estimated value:", estimate[i])
                print("[*]============================")

        print("Optimization Finished!")

        # Test model
        # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        predicted_vals = sess.run(pred, feed_dict={x: x_big})
        # Calculate accuracy
        accuracy = sess.run(good, feed_dict={x: x_big, y: y_big})
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy)

        plt.figure(1)
        ax = plt.subplot(211)
        ax2 = plt.subplot(212)
        ax.scatter(y_bigtest, pred, s=1)
        ax2.scatter(range(len(losses)), losses)
        ax.plot([y_bigtest.min(), y_bigtest.max()], [y_bigtest.min(), y_bigtest.max()], 'k--', lw=1)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        plt.show()

neuralnetwork()