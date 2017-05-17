
""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import DataExtractor
from tensorflow.contrib import learn
import pickle

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


max_entityTypes = 42
Cv_filepath = "resources/test_web_freepal"
TrainDatapath = "resources/train_web_freepal"
TestDataPath = "resources/test_web_freepal"

# import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# n_samples = mnist.train.num_examples

# ==================================================
x_text_train, y_train = DataExtractor.load_data_and_labels_new(TrainDatapath)
x_text_cv, y_dev = DataExtractor.load_data_and_labels_new(Cv_filepath)
# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text_train])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))
x_dev = np.array(list(vocab_processor.fit_transform(x_text_cv)))
print("Loading data...")
n_samples = len(y_train)
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train : {:d}".format(n_samples))

file = open("resources/rel_pickle_id", 'rb')
id_rel_Map = pickle.load(file)
file.close()


# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
# n_input = 784 # MNIST data input (img shape: 28*28)
n_input = x_train.shape[1]

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
X = tf.nn.sigmoid(X)

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X
batches = DataExtractor.batch_iter(list(zip(x_train, y_train)), batch_size=batch_size,
                                   num_epochs=training_epochs)
batches_test = DataExtractor.batch_iter(list(zip(x_dev, y_dev)), batch_size= 1,
                                   num_epochs=len(x_dev))

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    for i, batch in enumerate(batches):
        x_batch, y_batch = zip(*batch)
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Fit training using batch data
        _, cost_ = sess.run([optimizer, cost], feed_dict={X: x_batch})
        # Compute average loss
        avg_cost += cost_ / n_samples * batch_size

        # Display logs per epoch step
        # if batch % 250 == 0:
        print("Epoch:", '%04d' % (i + 1),
              "cost=", "{:.9f}".format(avg_cost))
    print("avg cost"+str(avg_cost))
    print("Optimization Finished!")

    # Applying encode and decode over test set

    init_threshold = 0.5*avg_cost
    in_count =0
    # test_data = x_train
    test_data = x_dev
    max_threshold = 5*avg_cost
    threshold = init_threshold
    while threshold < max_threshold:
        for test_x in enumerate(test_data):
            # test_x, test_y = zip(*batch)
            test = np.reshape(test_x[1], newshape= [1, len(x_train[1])])
            cost_test = sess.run(
                cost, feed_dict={X: test})

            if cost_test < threshold:
                in_count+=1
        threshold+= 0.25

    print("IN-DOMAIN : "+str(in_count))
    print ("Total : "+str(len(test_data)))
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(test_x[i], (1, x_train[1])))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (1, x_train[1])))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()