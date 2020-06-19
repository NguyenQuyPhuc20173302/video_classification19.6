import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


def huber_loss(delta_):
    huber = tf.keras.losses.Huber(
        delta=delta_, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss'
    )
    return huber

