#!/usr/bin/env python
# coding: utf-8

"""
Predictor Training
==================

To maximize accuracy, the yield predictor is periodically re-trained on encoder output
during encoder training. However, it is beneficial to seed the predictor by training it on random sequences.
"""

import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cupyck
import primo.models
import primo.tools.sequences as seqtools

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a yield predictor model.")
    parser.add_argument("--num_pairs", type=int, default=5000, help="Number of random sequence pairs.")
    parser.add_argument("--dna_length", type=int, default=80, help="Length of each DNA sequence.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--model_path", type=str, default="/tf/primo/data/models/yield-predictor.h5", help="Path to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    return parser.parse_args()

def main():
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    """Main function to train the predictor."""
    args = parse_args()
    
    # Initialize the CuPyCK session to manage GPU memory allocation before TensorFlow is loaded
    cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=1024, nthreads=128)
    
    # Initialize the hybridization simulator using CuPyCK
    simulator = primo.models.Simulator(cupyck_sess)
    
    # Generate a dataset of random sequence pairs with varying Hamming distances
    random_pairs, mut_rates = seqtools.random_mutant_pairs(args.num_pairs, args.dna_length)
    
    # Create a DataFrame to store the sequence pairs
    seq_pairs = pd.DataFrame({
        "target_features": random_pairs[:, 0],
        "query_features": random_pairs[:, 1]
    })
    
    # Simulate hybridization yields for the sequence pairs using NUPACK/CuPyCK
    sim_results = simulator.simulate(seq_pairs)
    
    del simulator, cupyck_sess
    # Initialize a yield predictor model
    predictor = primo.models.Predictor()
    
    # Convert sequence pairs to one-hot encoding format for model input
    onehot_seq_pairs = predictor.seq_pairs_to_onehots(seq_pairs)
    dataset = tf.data.Dataset.from_tensor_slices((onehot_seq_pairs, sim_results.duplex_yield))

    validation_split=0.2
    train_size = int((1-validation_split)*args.num_pairs)
    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Train the predictor using simulated yields as labels
    predictor.train(
        train_dataset, 
        val_dataset,
        epochs=args.epochs, 
        data_size = args.num_pairs,
    )
    
    # Save the trained predictor model for future use
    predictor.save(args.model_path)
    
    print(f"Model training completed and saved successfully at {args.model_path}.")

if __name__ == "__main__":
    main()
