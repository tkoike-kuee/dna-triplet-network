#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import primo.models
import cupyck
from tqdm import tqdm
import os
import argparse

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-t", "--target", type=str, help="target path")
    parse.add_argument("-q", "--query", type=str, help="query path")
    parse.add_argument("-o", "--output", type=str, help="output path")

    args = parse.parse_args()

    print("Loading the sequences")
    print("target: {}".format(args.target))
    print("query: {}".format(args.query))

    # Initialize the simulator
    cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=2048, nthreads=512)
    simulator = primo.models.Simulator(cupyck_sess)

    # Load the sequences
    target_seqs = pd.read_hdf(args.target, key='df', mode='r')
    query_seqs = pd.read_hdf(args.query, key='df', mode='r')
    total_store = pd.DataFrame()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Run the simulations in each query sequence
    for que_in in range(len(query_seqs.index)):
        pairs = (target_seqs
        .rename(columns={'FeatureSequence':'target_features'})
        .assign(query_features = query_seqs.iloc[int(que_in)].FeatureSequence)
        )
        print("label {}:".format(int(que_in)))

    # 4,000 here is just a memory-management batch size so that each progress chunk reports period of time.
        split_size = 4000
        nsplits = len(pairs) / split_size
        splits = np.array_split(pairs, nsplits)

        result_store = pd.DataFrame()
        try:
            for split in tqdm(splits, leave=False):
                results = simulator.simulate(split)
                result_store=pd.concat([result_store,results[['duplex_yield']]],axis=0)
        finally:
            total_store=pd.concat([total_store,result_store.rename(columns={"duplex_yield":"{}".format(que_in)})], axis=1)
            del result_store

    # Save the results
    total_store[total_store>1.0] = 1
    total_store.to_hdf(args.output, key='df', mode='w')


if __name__ == "__main__":
    main()