__author__ = 'mohamed'
import numpy as np
import tensorflow as tf

#1st) build the computation graph
x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.mul(x1,x2)
print(result)

#2nd) build what is supposed to happen in the session
with tf.Session() as sess:
    output = sess.run(result)

print(output)

