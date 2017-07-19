from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn import datasets, linear_model
from sklearn import model_selection
import numpy as np

boston = learn.datasets.load_dataset('boston')
x, y = boston.data, boston.target
# print(len(x), x[:10])
# print(len(y), y[:10])
y = np.reshape(y, [y.shape[0], 1])
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42)

total_len = X_train.shape[0]

# Parameters
learning_rate = 0.001
training_epochs = 5000
batch_size = 100
display_step = 10
# dropout_rate = 0.9
# Network Parameters
n_hidden_1 = 13  # 1st layer number of features
n_hidden_2 = 26  # 2nd layer number of features
n_hidden_3 = 1
n_hidden_4 = 1
n_input = X_train.shape[1]
n_classes = 1

# tf Graph input
x = tf.placeholder("float", [None, 13])
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

    mse = tf.reduce_mean(tf.square(pred - Y_test))
    squares = tf.reduce_sum(tf.square(tf.subtract(Y_test, tf.reduce_mean(Y_test))))
    good = tf.subtract(1.0, tf.divide(mse, tf.cast(squares, tf.float32)))

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.square(pred - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Training cycle
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
                avg_cost += c / total_batch

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
        predicted_vals = sess.run(pred, feed_dict={x: X_test})
        # Calculate accuracy
        accuracy = sess.run(good, feed_dict={x: X_test, y: Y_test})
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy)

        fig, ax = plt.subplots()
        ax.scatter(Y_test, predicted_vals, s=1)
        # ax2.scatter(range(len(losses)), losses)
        ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=1)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        # ax2.set_xlabel('Epoch')
        # ax2.set_ylabel('Loss')
        plt.show()

neuralnetwork()
