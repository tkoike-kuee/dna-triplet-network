import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
import random
from .dataset import Dataset
from tqdm import tqdm

class OpenImagesTrain(Dataset):
    def __init__(self, path, train, switch_every = 1000, model=None):
        self.path = path
        self.switch_every = switch_every
        self.model = model
        self.train=train

    def get_images(self, img_ids):
        image_dir = os.path.join(self.path, 'images')

        for img_id in img_ids:
            img_path = os.path.join(
                image_dir, '%s/%s.jpg' % (img_id[:2], img_id)
            )
            image = Image.open(img_path)
            image.load()
            yield image

    def random_pairs(self, batch_size):

        feature_dir = os.path.join(self.path, 'features')
        if self.model != None:
            feature_dir = os.path.join(feature_dir, self.model)
        # files = os.listdir(feature_dir)
        # files = [feature_dir+"/train.h5"]
        files=[self.train]
        while True:
            if(len(files) > 1):
                f_a, f_b = np.random.choice(files, 2, replace=False)
                sys.stdout.write("switching to %s and %s\n" % (f_a, f_b))

                df1 = pd.read_hdf(os.path.join(feature_dir, f_a))
                df2 = pd.read_hdf(os.path.join(feature_dir, f_b))

                df = pd.concat([df1, df2])
                n = len(df)
            else:
                df = pd.read_hdf(os.path.join(feature_dir, files[0]))
                n = len(df)
            for _ in range(self.switch_every):

                pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)

                yield df.index.values[pairs], df.values[pairs]

    def makeDataset(self, batch_size):
        feature_dir = os.path.join(self.path, 'features')
        if self.model != None:
            feature_dir = os.path.join(feature_dir, self.model)
        df = pd.read_hdf(os.path.join(feature_dir, self.train))

        x=df.values
        y=np.array([int(tbl_index.split("_")[-1]) for tbl_index in df.index]).reshape(-1,1)
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x)*2).batch(batch_size,drop_remainder=True)


    def makePairs(self, num_classes=10):
        feature_dir = os.path.join(self.path, 'features')
        if self.model != None:
            feature_dir = os.path.join(feature_dir, self.model)
        df = pd.read_hdf(os.path.join(feature_dir, self.train))

        x = df.values
        y = np.array([int(tbl_index.split("_")[-1]) for tbl_index in df.index])

        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]
            
            labels += list([1])
            pairs += [[x1, x2]]
        
            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]
            
            labels += list([0])
            pairs += [[x1, x2]]
            
        return np.array(pairs), np.array(labels)
                
class OpenImagesVal(Dataset):
    
    def __init__(self, val_path, model = None):
        feature_path = os.path.join(val_path) #, 'features')
        if model != None:
            feature_path = os.path.join(feature_path, model)
            feature_path = os.path.join(feature_path, '{}-validation.h5'.format(model))
        self.df = pd.read_hdf(feature_path)
        print(self.df)
        
    def random_pairs(self, batch_size):
        n = len(self.df)
        while True:
            pairs = np.random.permutation(n)[:batch_size*2].reshape(-1,2)
            yield self.df.index.values[pairs], self.df.values[pairs]
    
    def makeDataset(self,batch_size):
        x=self.df.values
        y=np.array([int(tbl_index.split("_")[-1]) for tbl_index in self.df.index]).reshape(-1,1)
        return tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=len(x)*2).batch(batch_size,drop_remainder=True)

    def makePairs(self, num_classes=10):
        x = self.df.values
        y = np.array([int(tbl_index.split("_")[-1]) for tbl_index in self.df.index])

        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = list()
        labels = list()

        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]
            
            labels += list([1])
            pairs += [[x1, x2]]
        
            label2 = random.randint(0, num_classes-1)
            while label2 == label1:
                label2 = random.randint(0, num_classes-1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]
            
            labels += list([0])
            pairs += [[x1, x2]]
            
        return np.array(pairs), np.array(labels)