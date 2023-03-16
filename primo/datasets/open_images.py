import os
import sys
import numpy as np
import pandas as pd

from PIL import Image

from .dataset import Dataset

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