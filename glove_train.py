import tensorflow as tf
# from var import VariationalAutoencoder
# from var_lstm import VariationalAutoencoder
import DataExtractor
from tensorflow.contrib import learn
import numpy as np
import pickle


max_entityTypes = 42
Cv_filepath = "resources/test_web_freepal"
TrainDatapath = "resources/train_web_freepal"
TestDataPath = "resources/test_web_freepal"
out_domain = "/home/mvidyasa/Downloads/outDomain_Manish.txt"

# import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# n_samples = mnist.train.num_examples

# ==================================================
x_text_train, y_train, max_document_length = DataExtractor.load_data_and_labels_glove(TrainDatapath)
x_text_cv, y_dev, _ = DataExtractor.load_data_and_labels_glove(Cv_filepath)
x_od, y_od , _= DataExtractor.load_data_and_labels_glove_custom(out_domain)
# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text_train])
# max_document_length = DataExtractor.g.max_sequence_length
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))
# x_dev = np.array(list(vocab_processor.fit_transform(x_text_cv)))
x_train = x_text_train
x_dev = x_text_cv
print("Loading data...")
n_samples = len(y_train)
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train : {:d}".format(n_samples))

# file = open("resources/rel_pickle_id", 'rb')
# id_rel_Map = pickle.load(file)
# file.close()

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
         ip_size = 14*300, #placeholder
         n_input= 2*VariationalAutoencoder.lstm_hidden_size*14, # Input to VAE
         n_z=2)  # dimensionality of latent space

f = open("Result_Log","w")
# Generate batches
batches = DataExtractor.batch_iter(list(zip(x_train, y_train)), batch_size= batch_size, num_epochs=10)
with tf.Session() as sess:
    vae_2d = train(network_architecture,learning_rate= learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(vae_2d.cost)
    sess.run(tf.global_variables_initializer())
    for i, batch in enumerate(batches):
        if i ==10:
            print "debug"
        x_batch, y_batch = zip(*batch)
        batch_size = len(x_batch)
        avg_cost = 0.
        # total_batch = int(n_samples / batch_size)
        # # Loop over all batches
        # for i in range(total_batch):
        #     batch_xs, _ = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        _, cost = sess.run([opt, vae_2d.cost], feed_dict= {vae_2d.ip : x_batch, vae_2d.batch_size : batch_size})
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        # if batch % 250 == 0:
        # print("Epoch:", '%04d' % (i+1),
        #       "cost=", "{:.9f}".format(avg_cost))
        print("Epoch:" + str(i + 1)+
              "cost="+ str(avg_cost))
    ######## TEST #######
    print "Average Cost : "+str(avg_cost)
    f.write("Average Cost : "+str(avg_cost))
    threshold = 0.085
    # Applying encode and decode over test set

    in_count = 0
    test_data = x_train
    # test_data = x_dev
    # max_threshold = 5 * avg_cost
    for i,test_x in enumerate(test_data):
        # test_x, test_y = zip(*batch)

        test = np.reshape(test_x, newshape=[1, len(x_train[1])])
        cost_test = sess.run(vae_2d.cost, feed_dict={ vae_2d.ip: test, vae_2d.batch_size: 1})
        print "c1: "+str(i)+" : "+str(cost_test)
        f.write("c1: "+str(i)+" : "+str(cost_test)+"\n")
        if cost_test < threshold:
            in_count += 1
    train_in_count = in_count
    print("IN-DOMAIN : " + str(in_count))
    f.write("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data)))
    f.write("Total : " + str(len(test_data)))

    ######## TEST #######
    print "Average Cost : " + str(avg_cost)
    # threshold = avg_cost
    # Applying encode and decode over test set

    in_count = 0
    test_data = x_dev
    # test_data = x_dev
    # max_threshold = 5 * avg_cost
    for i,test_x in enumerate(test_data):
        # test_x, test_y = zip(*batch)
        test = np.reshape(test_x, newshape=[1, len(x_train[1])])
        cost_test = sess.run(vae_2d.cost, feed_dict={vae_2d.ip: test, vae_2d.batch_size: 1})
        print "c2: "+str(i)+" : " + str(cost_test)
        f.write("c2: " + str(i) + " : " + str(cost_test)+"\n")
        if cost_test < threshold:
            in_count += 1
    test_in_count = in_count
    print("IN-DOMAIN : " + str(in_count))
    f.write("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data)))
    f.write("Total : " + str(len(test_data)))

    ######## TEST #######
    print "Average Cost : " + str(avg_cost)
    # threshold = avg_cost
    # Applying encode and decode over test set

    in_count = 0
    test_data = x_od
    # test_data = x_dev
    # max_threshold = 5 * avg_cost
    for i,test_x in enumerate(test_data):
        # test_x, test_y = zip(*batch)
        test = np.reshape(test_x, newshape=[1, len(x_train[1])])
        cost_test = sess.run(vae_2d.cost, feed_dict={vae_2d.ip: test, vae_2d.batch_size: 1})
        print ("c3: "+str(i)+" : " + str(cost_test))
        f.write("c3: " + str(i) + " : " + str(cost_test)+"\n")

        if cost_test < threshold:
            in_count += 1
    od_in_count = in_count
    print("IN-DOMAIN : " + str(in_count))
    f.write("IN-DOMAIN : " + str(in_count))
    print ("Total : " + str(len(test_data)))
    f.write("Total : " + str(len(test_data)))
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(test_x[i], (1, x_train[1])))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (1, x_train[1])))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()
    print "Train In Domain / Total: "+str(train_in_count) +" / " + str(len(x_train))
    f.write("Train In Domain / Total: "+str(train_in_count) +" / " + str(len(x_train)))
    print "Test In Domain / Total: " + str(test_in_count) +" / " + str(len(x_dev))
    f.write("Test In Domain / Total: " + str(test_in_count) +" / " + str(len(x_dev)))
    print "OD In Domain / Total: " + str(od_in_count) +" / " + str(len(x_od))
    f.write("OD In Domain / Total: " + str(od_in_count) +" / " + str(len(x_od)))

