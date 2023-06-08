#!/usr/bin/env python
# coding: utf-8

# Encoder Training
# ==============

# In[1]:


# メモリの調節
import os


# In[2]:


#get_ipython().run_line_magic('pylab', 'notebook')

import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow_addons.losses as tfa_loss
from primo.tools.losses import Losses
import primo.models
import primo.datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cupyck
import sys
import time
# import statistics
from tqdm import tqdm
from sklearn import metrics
import gc
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.3, y_true.dtype)))

def primo_train(encoder, encoder_train_batch_size):
    pairs_train, labels_train = train_dataset.makePairs(num_classes=num_classes)
    pairs_val, labels_val = val_dataset.makePairs(num_classes=num_classes)
    yield_predictor = primo.models.Predictor(model_path="/home/work/tkoike/primo-classification/models/predictor-models-len80.h5")
    encoder_trainer = primo.models.EncoderTrainer(encoder, yield_predictor)
    encoder_trainer.model.compile(tf.keras.optimizers.Adagrad(1e-2), loss='binary_crossentropy', metrics=['accuracy'])
    history = encoder_trainer.model.fit(
    [pairs_train[:,0],pairs_train[:,1]],
    labels_train,
    batch_size=128,
    epochs = 150, 
    validation_data = ([pairs_val[:,0],pairs_val[:,1]],labels_val),
    verbose = 1
    )
    return history, encoder

def triplet_train(encoder,encoder_train_batch_size):
    margin=0.8
    train_data = train_dataset.makeDataset(encoder_train_batch_size)
    val_data = val_dataset.makeDataset(encoder_train_batch_size)
    
    triplet_network=primo.models.TripletNetwork(encoder, encoder_train_batch_size, margin=margin)
    loss=tfa_loss.TripletSemiHardLoss(margin=margin, distance_metric=Losses(margin=margin,seq_len=80,batch=encoder_train_batch_size).make_triplet)
    triplet_network.model.compile(tf.keras.optimizers.Adagrad(1e-2), loss=loss)
    history = triplet_network.model.fit(
    train_data,
    batch_size=encoder_train_batch_size,
    epochs = 150,
    validation_data = val_data,
    verbose = 1
    )
    return history, encoder

def main():
    global num_classes, train_dataset, val_dataset
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    data_dir = ""
    save_dir=""
    dataset_name=input("dataset:")
    num_classes=10
    cnn_model = input("model:")

    loss_choice = int(input("loss fucntion(bce: 0,Triplet: 1):"))

    length=80
    data_dir+=dataset_name+"/"
    save_dir+=dataset_name+"/"
    enc_name = input("encoder name")
    seq_name = input("DNA sequences name")
    out_name=input("out name:")
    acc_list = []
    speed_list = []
    

    
    train_dataset = primo.datasets.OpenImagesTrain(
    data_dir+'train/', switch_every=10**5, train=data_dir+"train/features/"+cnn_model+"/train.h5", model=None
    )

    val_dataset = primo.datasets.OpenImagesVal(data_dir+'test/features/{}/test.h5'.format(cnn_model), model=None)
    query_features = pd.read_hdf(os.path.join(data_dir,'queries/{}/features.h5'.format(cnn_model)))
    target_features = pd.read_hdf(data_dir+'test/features/{}/test.h5'.format(cnn_model))


    encoder_train_batch_size = 128

    encoder = primo.models.Encoder(out_len=length)
    if(loss_choice == 0):
        history, encoder = primo_train(encoder,encoder_train_batch_size)
    elif(loss_choice == 1):
        history, encoder = triplet_train(encoder,encoder_train_batch_size)
    if not os.path.exists(save_dir+'models/{}'.format(cnn_model)):
        os.makedirs(save_dir+'models/{}/'.format(cnn_model))


    query_seqs = encoder.encode_feature_seqs(query_features)
    query_seqs = pd.DataFrame(
    query_seqs, index=query_features.index, columns=['FeatureSequence']
    )

    target_seqs = encoder.encode_feature_seqs(target_features)
    target_seqs = pd.DataFrame(
        target_seqs, index=target_features.index, columns=['FeatureSequence']
    )
    pd.DataFrame(
        query_seqs, index=query_features.index, columns=['FeatureSequence']
    ).to_hdf(
        os.path.join(save_dir,'queries/{}/{}.h5'.format(cnn_model, seq_name)), key='df', mode='w'
    )
    pd.DataFrame(
        target_seqs, index=target_features.index, columns=['FeatureSequence']
    ).to_hdf(
        os.path.join(save_dir,'targets/{}/{}.h5'.format(cnn_model, seq_name)), key='df', mode='w'
    )
    encoder.save(save_dir+'models/{}/{}.h5'.format(cnn_model,enc_name))
    
if __name__=="__main__":
    main()