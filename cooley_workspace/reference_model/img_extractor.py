try:
    import caffe
except:
    k=0

import numpy as np
from PIL import Image
import cPickle as pickle
import scipy.misc as msc
import matplotlib.pyplot as plt
import os
import datetime

def getImgNP(cifar_data, idx):
    """Gets a 2D Numpy RGB Matrix of the Image"""
    red = cifar_data['data'][idx][:1024].reshape(32, 32)
    green = cifar_data['data'][idx][1024:2048].reshape(32, 32)
    blue = cifar_data['data'][idx][2048:].reshape(32, 32)
    im = np.zeros((32, 32, 3))
    im[:, :, 0] = red
    im[:, :, 1] = green
    im[:, :, 2] = blue
    return im

def saveImg(im, fname='image.jpeg'):
    """Saves input numpy array containing the image as JPEG image onto HDD"""
    msc.imsave(fname, im)

def viewImg(cifar_data, idx):
    im = plt.imshow(getImgNP(cifar_data, idx))
    plt.show()

def oneHot(labels, numClasses=10):
    arr = np.zeros([labels.shape[0], numClasses])
    for i in xrange(arr.shape[0]):
        arr[i, labels[i]] = 1
    return arr

def dumpImgs(cifar_data, mode):
    train_root = r'images' + os.sep
    names = cifar_data['filenames']
    f_txt = open('imgs_' + mode + '.txt', 'ab')
    for i in xrange(len(names)):
        path = train_root + names[i]
        print 'writing', names[i]
        saveImg(getImgNP(cifar_data, i), fname=path)
        f_txt.write(path + ' ' + str(cifar_data['labels'][i]) + '\n')
    f_txt.close()

def transform_data(cifar_data):
    d = cifar_data['data']
    i = 0
    for p in d:
        d[i] = (p - p.mean()) / np.max(p.std(), 1./np.sqrt(p.shape[0]))
        i += 1


def extract_images(name, mode):

    data = pickle.load(open(r'data' + os.sep + name, 'rb'))
    dumpImgs(data, mode)


if __name__ == "__main__":
    for i in xrange(1, 6):
        extract_images('data_batch_' + str(i), 'train')

    extract_images('test_batch', 'test')