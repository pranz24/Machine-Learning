import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-2, 2, 101).astype(np.float32)
y_data = x_data**2 + np.random.randn(*x_data.shape) * 0.5
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(np.random.randn())
W2 = tf.Variable(np.random.randn())
W3 = tf.Variable(np.random.randn())
weights = tf.Variable(np.random.randn())
baises = tf.Variable(np.random.randn())



# Parameters
rate = 0.0009
iteration = 500

# Model
model = tf.mul(X**4, W) + tf.mul(X**3, weights) + tf.mul(X**2, W2) + tf.mul(X, W3) + baises
#Cost Function
loss = tf.square(Y - model)
#Optimizer
training_OP = tf.train.GradientDescentOptimizer(rate).minimize(loss)
#tensorflow session run
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.initialize_all_variables().run()

    for i in range(iteration):
        for (x, y) in zip(x_data, y_data):
            sess.run(training_OP, feed_dict={X: x, Y: y})

    print sess.run(loss, feed_dict={X: x, Y: y})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, sess.run(W) * x_data**4 + sess.run(weights) * x_data**3 + sess.run(W2) * x_data**2 + sess.run(W3) * x_data + sess.run(baises))
    plt.legend()
    plt.show()