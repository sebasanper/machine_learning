"""
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)

optimisation function (optimiser) > minimise cost (AdamOptimizer ... Gradient Descent, AdaGrad, ...)

backpropagation

feed forward + backpropagation = epoch
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

"""
10 classes, 0-9

0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
...
9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
"""
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# matrix: height x weight
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):

    # (input_data * weights) + biases

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
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
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    #                                            learning_rate = 0.001
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    # optimiser = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + back propagation
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                x_epoch, y_epoch = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimiser, cost], feed_dict={x: x_epoch, y: y_epoch})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'Loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
