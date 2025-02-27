#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras import backend as K
from tqdm import tqdm
from primo.tools.losses import Losses
from primo.tools import sequences as seqtools
import os
import tensorflow as tf
import tensorflow_addons.losses as tfa_loss
import primo.models
import primo.datasets
import pandas as pd
import argparse

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

def triplet_train(train_dataset, val_dataset, encoder, batch_size, seq_len=80, epochs=150, margin=0.8):
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

def encode_queries(encoder, train_data, num_classes):
    train_features = pd.read_hdf(train_data)
    query_seqs = []

    for i in tqdm(range(num_classes), desc="Encoding queries"):
        group_dna = train_features[train_features.index.str.endswith(f"_{i}")]
        train_seqs = encoder.encode_feature_seqs(group_dna)
        dna_seqs = seqtools.seqs_to_onehots(train_seqs).sum(0).reshape(1, -1, 4)
        query_seqs.append(seqtools.onehots_to_seqs(dna_seqs))
    return query_seqs

def main():
    # Set GPU memory growth
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_data", type=str, help="Path to the training data")
    parse.add_argument("--test_data", type=str, help="Path to the test data")
    parse.add_argument("--target_seqs", type=str, help="Output path for the target sequences")
    parse.add_argument("--query_seqs", type=str, help="Output path for the query sequences")
    parse.add_argument("--encoder", type=str, help="Path to the encoder")
    parse.add_argument("--predictor", type=str, help="Path to the predictor")
    parse.add_argument("-l", "--length", type=int, help="Length of the feature sequences", default=80)
    parse.add_argument("--loss", type=int, help="Loss function (0: BCE, 1: Triplet)", default=1)
    parse.add_argument("-n", "--num_classes", type=int, help="Number of classes", default=10)
    parse.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parse.add_argument("--epochs", type=int, help="Number of epochs", default=150)
    parse.add_argument("--margin", type=float, help="Margin for triplet loss", default=0.8)

    args = parse.parse_args()
    
    # Initialize the datasets 
    train_dataset = primo.datasets.OpenImagesTrain(args.train_data)
    val_dataset = primo.datasets.OpenImagesVal(args.test_data)

    encoder = primo.models.Encoder(out_len=args.length)
    print("Training the encoder")
    if(args.loss == 0):
        _, encoder = primo_train(train_dataset, val_dataset, args.predictor, encoder,args.batch_size, args.num_classes, args.epochs)
    elif(args.loss == 1):
        _, encoder = triplet_train(train_dataset, val_dataset, encoder, args.batch_size, args.length, args.epochs, args.margin)
    
    os.makedirs(os.path.dirname(args.encoder), exist_ok=True)

    print("Encoding the sequences")
    # encode the features into DNA sequences
    target_features = pd.read_hdf(args.test_data)
    target_seqs = encoder.encode_feature_seqs(target_features)
    query_seqs = encode_queries(encoder, args.train_data, args.num_classes)

    print("Saving the sequences")
    # Save the DNA sequences
    pd.DataFrame(query_seqs, index=[f"label_{i}" for i in range(args.num_classes)], columns=['FeatureSequence']).to_hdf(args.query_seqs, key='df', mode='w')
    pd.DataFrame(target_seqs, index=target_features.index, columns=['FeatureSequence']).to_hdf(args.target_seqs, key='df', mode='w')
    # Save the encoder
    encoder.save(args.encoder)
    
if __name__=="__main__":
    main()