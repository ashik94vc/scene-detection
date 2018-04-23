import numpy as np
from PIL import Image
import six.moves.cPickle as pickle

wnids = list(map(lambda x: x.strip(), open('wnids.txt').readlines()))

data = {}
data['train'] = {}
data['train']['data'] = np.ndarray(shape=(100000, 3, 64, 64), dtype=np.uint8)
data['train']['target'] = np.zeros(shape=(100000,200), dtype=int)
data['val'] = {}
data['class_index'] = {}
data['val']['data'] = np.ndarray(shape=(10000, 3, 64, 64), dtype=np.uint8)
data['val']['target'] = np.zeros(shape=(10000,200), dtype=int)

for i in range(len(wnids)):
    wnid = wnids[i]
    print("{}: {} / {}".format(wnid, i + 1, len(wnids)))
    for j in range(500):
        path = "train/{0}/images/{0}_{1}.JPEG".format(wnid, j)
        data['train']['data'][i * 500 + j] = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)
        data['train']['target'][i * 500 + j][i] = 1
        data['class_index'][i] = wnid


for i, line in enumerate(map(lambda s: s.strip(), open('val/val_annotations.txt'))):
    name, wnid = line.split('\t')[0:2]
    path = "val/images/{0}".format(name)
    data['val']['data'][i] = np.asarray(Image.open(path).convert('RGB')).transpose(2, 0, 1)
    data['val']['target'][i][wnids.index(wnid)] = 1

print("Dump to train.pkl...")
pickle.dump(data, open('tiny_imagenet.pkl', 'wb', -1))
