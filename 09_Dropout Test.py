from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("MNIST", one_hot=True, reshape=False)

# Parameters
learning_rate = 0.0005
training_epochs = 50
batch_size = 256  # Decrease batch size if you don't have enough memory
test_size = 512
display_step = 5
keep_probability = 0.5

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

n_hidden_layer = 256 # layer number of features

x_test = mnist.test.images
y_test = mnist.test.labels

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'output_layer': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'output_layer': tf.Variable(tf.random_normal([n_classes]))
}

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

keep_prob = tf.placeholder(tf.float32)

# Hidden layer with RELU activation
hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

# Output layer with linear activation
logits = tf.matmul(hidden_layer, weights['output_layer']) + biases['output_layer']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x,
                                           y: batch_y,
                                           keep_prob: keep_probability})
        # Display logs per epoch step
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict={x: batch_x,
                                          y: batch_y,
                                          keep_prob: keep_probability})
            validation_accuracy = sess.run(accuracy, feed_dict={x: x_test[:test_size],
                                                                y: y_test[:test_size],
                                                                keep_prob: 1})
            print("Epoch:", '%04d' % (epoch+1),
                  "Cost = {:.9f}".format(c),
                  "Validation Accuracy = {:.9f}".format(validation_accuracy))

    print("Optimization Finished!")
