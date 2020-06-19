import pickle
import tensorflow as tf
import tensorflow as tf

tf.debugging.set_log_device_placement(True)
try:
    with tf.device('/device:GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
except RuntimeError as e:
    print(e)
with open('/data/phucnq/data/video_classification/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('/data/phucnq/data/video_classification/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

x_train = x_train.reshape(61607, 1, 2048)

import model as m

model = m.lstm((1, 2048), 101)
model.fit(x_train, y_train, epochs=100, batch_size=32)
