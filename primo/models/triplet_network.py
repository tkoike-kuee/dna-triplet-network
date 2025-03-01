import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import pandas as pd

class TripletNetwork:

    def __init__(self, encoder, batch=128, margin=1.0):

        self.encoder = encoder
        self.batch = batch
        self.margin=margin

        # Essentially, we started with a batch of feature-vector pairs...
        # ...And turned them into a pair of feature-vector batches.
        X = layers.Input(shape=(encoder.input_dim,), name="input_data", batch_size=self.batch)

        # Independently transforms the batches of feature vectors into soft-max encoded DNA sequences.
        S1 = encoder(X)
        S1_onehots = layers.Lambda(lambda S: S*tf.one_hot(tf.math.argmax(S,2),4))(S1) #transforms one-hot vector
        self.model = tf.keras.Model(inputs=X, outputs=S1_onehots)

        self.model.summary()
