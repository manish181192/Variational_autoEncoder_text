import tensorflow as tf
from var import VariationalAutoencoder
import DataExtractor
from tensorflow.contrib import learn
import numpy as np
import pickle


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

max_accuracy = 100
batch_size = 256
learning_rate = 0.001
training_epochs = 75
def train(network_architecture, learning_rate=learning_rate,
          batch_size=batch_size):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # # Training cycle
    # for epoch in range(training_epochs):
    #     avg_cost = 0.
    #     total_batch = int(n_samples / batch_size)
    #     # Loop over all batches
    #     for i in range(total_batch):
    #         batch_xs, _ = mnist.train.next_batch(batch_size)
    #
    #         # Fit training using batch data
    #         cost = vae.partial_fit(batch_xs)
    #         # Compute average loss
    #         avg_cost += cost / n_samples * batch_size
    #
    #     # Display logs per epoch step
    #     if epoch % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch+1),
    #               "cost=", "{:.9f}".format(avg_cost))
    return vae

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=x_train.shape[1], # MNIST data input (img shape: 28*28)
         n_z=2)  # dimensionality of latent space


# Generate batches
batches = DataExtractor.batch_iter(list(zip(x_train, y_train)), batch_size= batch_size, num_epochs=75)
with tf.Session() as sess:
    vae_2d = train(network_architecture,learning_rate= learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(vae_2d.cost)
    sess.run(tf.global_variables_initializer())
    for i, batch in enumerate(batches):
        x_batch, y_batch = zip(*batch)
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # # Loop over all batches
        # for i in range(total_batch):
        #     batch_xs, _ = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        _, cost = sess.run([opt, vae_2d.cost], feed_dict= {vae_2d.x : x_batch})
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        # if batch % 250 == 0:
        print("Epoch:", '%04d' % (i+1),
              "cost=", "{:.9f}".format(avg_cost))

    ######## TEST #######

    threshold = avg_cost
    # Applying encode and decode over test set

    in_count = 0
    # test_data = x_train
    test_data = x_dev
    max_threshold = 5 * avg_cost
    for test_x in enumerate(test_data):
        # test_x, test_y = zip(*batch)
        test = np.reshape(test_x[1], newshape=[1, len(x_train[1])])
        cost_test = sess.run(vae_2d.cost, feed_dict={ vae_2d.cost: test})

        if cost_test < threshold:
            in_count += 1

    print("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data)))
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(test_x[i], (1, x_train[1])))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (1, x_train[1])))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()

