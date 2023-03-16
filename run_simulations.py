#!/usr/bin/env python
# coding: utf-8

# # Run Simulations
# 
# Given an encoded dataset of targets and queries, run simulations.

# In[1]:


import numpy as np
import pandas as pd

import primo.models

import cupyck

from tqdm import tqdm
import os

# In[2]:

#これは謎
#hosts = [
#    ("localhost", 2046),
#]
#client = cupyck.Client(hosts)
cupyck_sess = cupyck.GPUSession(max_seqlen=200, nblocks=1024, nthreads=128)
simulator = primo.models.Simulator(cupyck_sess)

cnn_model = input("model name:")
file=input("file:")
# In[3]:
dataset_name = input("dataset:")+"/"
import os
data_dir = '/home/work/tkoike/primo-classification/'
data_dir = data_dir+dataset_name
num_tab = int(input(".h5(0) or .npy(1):"))

if(num_tab==0):
    tar_name=input("dna sequences file name:")
    que_name=tar_name #input("query file name:")
    out_name=input("output file name:")
    target_seqs = pd.read_hdf(os.path.join(data_dir,'targets/{}/{}/{}.h5'.format(cnn_model,file,tar_name)))
    query_seqs = pd.read_hdf(os.path.join(data_dir,'queries/{}/{}/{}.h5'.format(cnn_model,file,que_name)))
if(num_tab==1):
    tar_name=input("dna sequences file name:")
    que_name=tar_name #input("query file name:")
    index_name=input("dna sequences index file name:")
    out_name=input("output file name:")
    q_index=np.load(os.path.join(data_dir,'queries/{}/{}/{}.npy'.format(cnn_model,file,index_name)))
    q_data=np.load(os.path.join(data_dir,'queries/{}/{}/{}.npy'.format(cnn_model,file,que_name)))
    t_index = np.load(os.path.join(data_dir,'targets/{}/{}/{}.npy'.format(cnn_model,file,index_name)))
    t_data = np.load(os.path.join(data_dir,'targets/{}/{}/{}.npy'.format(cnn_model,file,tar_name)))
    target_seqs=pd.DataFrame(t_data,index=t_index,columns=["FeatureSequence"])
    query_seqs=pd.DataFrame(q_data,index=q_index,columns=["FeatureSequence"])
# In[4]:
total_store = pd.DataFrame()
if not os.path.exists(data_dir+'simulation/{}'.format(cnn_model)):
    os.mkdir(data_dir+'simulation/{}'.format(cnn_model))
for que_in in range(len(query_seqs.index)):
    pairs = (target_seqs
    .rename(columns={'FeatureSequence':'target_features'})
    .assign(query_features = query_seqs.iloc[int(que_in)].FeatureSequence)
    )
    print("label {}:".format(int(que_in)))


# In[5]:


# 4,000 here is just a memory-management batch size so that each progress chunk reports period of time.
    split_size = 4000
    nsplits = len(pairs) / split_size
    splits = np.array_split(pairs, nsplits)


# In[6]:

    result_store = pd.DataFrame()
    try:
        for split in tqdm(splits, leave=False):
            results = simulator.simulate(split)
            result_store=pd.concat([result_store,results[['duplex_yield']]],axis=0)
    finally:
        total_store=pd.concat([total_store,result_store.rename(columns={"duplex_yield":"label_{}".format(que_in)})], axis=1)
        del result_store

total_store[total_store>1.0] = 1
total_store.to_hdf(os.path.join(data_dir,'simulation/{}/{}.h5'.format(cnn_model, out_name)), key='df', mode='w')
# In[ ]:




