#!/usr/bin/env python
# coding: utf-8
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons.losses as tfa_loss
from primo.tools.losses import Losses
import primo.models
import primo.datasets
import pandas as pd
import argparse

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.3, y_true.dtype)))

def primo_train(train_dataset, val_dataset, pred, encoder, batch_size, num_classes, epochs=150):
    pairs_train, labels_train = train_dataset.makePairs(num_classes=num_classes)
    pairs_val, labels_val = val_dataset.makePairs(num_classes=num_classes)
    yield_predictor = primo.models.Predictor(model_path=pred)
    encoder_trainer = primo.models.EncoderTrainer(encoder, yield_predictor)
    encoder_trainer.model.compile(tf.keras.optimizers.Adagrad(1e-2), loss='binary_crossentropy', metrics=['accuracy'])
    history = encoder_trainer.model.fit(
    [pairs_train[:,0],pairs_train[:,1]],
    labels_train,
    batch_size=batch_size,
    epochs = epochs, 
    validation_data = ([pairs_val[:,0],pairs_val[:,1]],labels_val),
    verbose = 1
    )
    return history, encoder

def triplet_train(train_dataset, val_dataset, encoder, batch_size, seq_len=80, epochs=150):
    margin=0.8
    train_data = train_dataset.makeDataset(batch_size)
    val_data = val_dataset.makeDataset(batch_size)
    
    triplet_network=primo.models.TripletNetwork(encoder, batch_size, margin=margin)
    loss=tfa_loss.TripletSemiHardLoss(margin=margin, distance_metric=Losses(margin=margin,seq_len=seq_len,batch=batch_size).make_triplet)
    triplet_network.model.compile(tf.keras.optimizers.Adagrad(1e-2), loss=loss)
    history = triplet_network.model.fit(
    train_data,
    batch_size=batch_size,
    epochs = epochs,
    validation_data = val_data,
    verbose = 1
    )
    return history, encoder

def main():
    # Set GPU memory growth
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    parse = argparse.ArgumentParser()
    parse.add_argument("-tr", "--train_data", type=str, help="Path to the training data")
    parse.add_argument("-te", "--test_data", type=str, help="Path to the test data")
    parse.add_argument("-qf", "--query_features", type=str, help="Path to the query features data")
    parse.add_argument("-tf", "--target_features", type=str, help="Path to the target features data")
    parse.add_argument("-ts", "--target_seqs", type=str, help="Output path for the target sequences")
    parse.add_argument("-qs", "--query_seqs", type=str, help="Output path for the query sequences")
    parse.add_argument("-en", "--encoder", type=str, help="Path to the encoder")
    parse.add_argument("-pred", "--predictor", type=str, help="Path to the predictor")
    parse.add_argument("-l", "--length", type=int, help="Length of the feature sequences", default=80)
    parse.add_argument("-lo", "--loss", type=int, help="Loss function (0: BCE, 1: Triplet)", default=1)
    parse.add_argument("-n", "--num_classes", type=int, help="Number of classes", default=10)
    parse.add_argument("-bs", "--batch_size", type=int, help="Batch size", default=128)
    parse.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=150)

    args = parse.parse_args()
    
    # Initialize the datasets 
    train_dataset = primo.datasets.OpenImagesTrain(args.train_data)
    val_dataset = primo.datasets.OpenImagesVal(args.test_data)
    query_features = pd.read_hdf(args.query_features)
    target_features = pd.read_hdf(args.target_features)

    encoder = primo.models.Encoder(out_len=args.length)
    if(args.loss == 0):
        _, encoder = primo_train(train_dataset, val_dataset, args.predictor, encoder,args.batch_size, args.num_classes, args.epochs)
    elif(args.loss == 1):
        _, encoder = triplet_train(train_dataset, val_dataset, encoder, args.batch_size, args.length, args.epochs)
    
    os.makedirs(os.path.dirname(args.encoder), exist_ok=True)

    # encode the features into DNA sequences
    query_seqs = encoder.encode_feature_seqs(query_features)
    query_seqs = pd.DataFrame(query_seqs, index=target_features.index, columns=['FeatureSequence'])
    target_seqs = encoder.encode_feature_seqs(target_features)
    target_seqs = pd.DataFrame(target_seqs, index=target_features.index, columns=['FeatureSequence'])

    # Save the DNA sequences
    pd.DataFrame(query_seqs, index=query_features.index, columns=['FeatureSequence']).to_hdf(args.query_seqs, key='df', mode='w')
    pd.DataFrame(target_seqs, index=target_features.index, columns=['FeatureSequence']).to_hdf(args.target_seqs, key='df', mode='w')
    # Save the encoder
    encoder.save(args.encoder)
    
if __name__=="__main__":
    main()