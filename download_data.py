import sys
import subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "gluonnlp==0.9.1"])

import gluonnlp as nlp

train_dataset, test_dataset = [nlp.data.IMDB(root='data/imdb', segment=segment)
                               for segment in ('train', 'test')]

import numpy as np
train_dataset=np.asarray(train_dataset)
test_dataset=np.asarray(test_dataset)

import os
os.makedirs("./output/model", exist_ok=True)
os.makedirs("./output/data", exist_ok=True)
os.makedirs("./data/train", exist_ok=True)
os.makedirs("./data/test", exist_ok=True)

np.save("./data/train/train.npy", train_dataset)
np.save("./data/test/test.npy", test_dataset)
