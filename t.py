from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

sess = tf.InteractiveSession()

# Import MNIST data

def load_clinical_eeg_data(datapath, sub):
  # input arguments:
  # datapath (string): path to the root directory
  # sub (string): subject ID (e.g. chb01, chb02, etc)
  
  # output:
  # eegdata (numpy array): samples x channels data matrix
  # eegevents (pandas dataframe): labels and chunks
  # channel_names (list): names of the channels
  import pandas as pd
  alldata = pd.read_csv(os.path.join(datapath, sub + '.csv'))
  alldata.rename(columns={'Unnamed: 0': 'Index'})
  eegevents = alldata[['labels', 'chunks']]
  alldata.drop(['Unnamed: 0', 'labels', 'chunks'], axis=1, inplace=True)
  names = alldata.keys()
  return alldata.iloc[:].as_matrix(), eegevents, names

data, label_chunk, nodes = load_clinical_eeg_data('train/','chb01')

labels_and_chunks = label_chunk.as_matrix()
s_res = labels_and_chunks[:,0]

print (data.shape)
print (s_res.shape)


# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 1000
display_step = 1

#partition data
import random
k = data.shape[0]

l = []
for i in range(0, 40):
  for n in range(0, 10000):
    l.append(i)

random.shuffle(l)
l = np.array(l)
print (l.shape)

test_data = []
test_s_res = []

for i in range(400001, data.shape[0]):
  test_data.append(data[i])
  test_s_res.append(s_res[i])

test_data = np.asarray(test_data)
test_s_res = np.asarray(test_s_res)

# Network Parameters
n_hidden_1 = 23 # 1st layer number of features
n_hidden_2 = 23 # 2nd layer number of features
n_input = 23 # EEG input
n_classes = 2 # either seziure or no seziure

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder(tf.int32, [None, ])


# Create model
def multilayer_perceptron(x, weights, biases):
  # Hidden layer with RELU activation
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.relu(layer_1)
  # Hidden layer with RELU activation
  layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  layer_2 = tf.nn.relu(layer_2)
  # Output layer with linear activation
  out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
  return out_layer

# Store layers weight & bias
weights = {
  'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
  'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
  'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
  'b1': tf.Variable(tf.random_normal([n_hidden_1])),
  'b2': tf.Variable(tf.random_normal([n_hidden_2])),
  'out': tf.Variable(tf.random_normal([n_classes]))
}

tf.global_variables_initializer().run()

# Construct model
pred = multilayer_perceptron(x, weights, biases)
print (pred)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
  sess.run(init)

  # Training cycle
  for epoch in range(training_epochs):
    ##generate boolean mask to separate 40 different training batches
    mask = [j == epoch for j in l]
    mask = np.array(mask)
    avg_cost = 0.
    total_batch = int(data.shape[0]/batch_size)
    # Loop over all batches
    for i in range(total_batch):
      batch_x, batch_y = data[mask], s_res[mask]
      batch_y = batch_y
      batch_x = batch_x
      # Run optimization op (backprop) and cost op (to get loss value)
      _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                            # Compute average loss
      avg_cost += c / total_batch
      # Display logs per epoch step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

  print("Optimization Finished!")
    
  # Test model
  correct_prediction = tf.equal(tf.argmax(pred, 0), tf.argmax(y, 0))
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy:", accuracy.eval({x: test_data, y: test_s_res}))
