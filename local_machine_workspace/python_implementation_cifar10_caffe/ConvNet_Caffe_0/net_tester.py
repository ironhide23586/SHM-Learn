try:
    import caffe
except:
    k=0

import numpy as np
from PIL import Image
import cPickle as pickle
import scipy.misc as msc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix



def save_heatmap(conf_mat, save_name, labels):
    fig, ax = plt.subplots()

    heatmap = ax.pcolor(conf_mat)
    ax.set_xticks(np.arange(conf_mat.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(conf_mat.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)

    ax.set_xlabel('Predicted')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('True')

    for y in range(conf_mat.shape[0]):
        for x in range(conf_mat.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.0f' % conf_mat[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
    plt.savefig(save_name)


def testModel(modelName):
    caffe.set_mode_gpu()
    net = caffe.Net('conv.prototxt', modelName, caffe.TEST)

    batch_size = 100

    f = open('imgs_test.txt', 'rb')
    t = f.readlines()

    test_size = len(t) / batch_size * batch_size

    true = np.array([int(e.strip().split(' ')[1]) for e in t])

    preds = []

    num_iters = test_size / batch_size

    for i in xrange(num_iters):
        print 'Batch', i+1, 'of', num_iters, 'Model Name:', modelName
        net.forward()
        preds += list(net.blobs['ip1'].data.argmax(axis=1))
    preds = np.array(preds)

    acc = accuracy_score(true[:test_size], preds)
    prec, rec, fsc, _ = precision_recall_fscore_support(true[:test_size], preds)
    conf_mat = confusion_matrix(true[:test_size], preds)

    f = open('stats_' + str(test_size) + '_' + modelName.split(os.sep)[-1] + '.txt', 'wb')

    f.writelines(['Accuracy = ' + str(acc) + '\n', 'Precision = ' + str(prec) + '\n', 'Recall = ' + str(rec) + '\n', 'F-Score = ' + str(fsc) + '\n', '\nConfusion Matrix :-\n', str(conf_mat)])
    print_results(acc, prec, rec, fsc)
    save_heatmap(conf_mat, 'confusion matrix.jpg', ['Airplane', 'Auto', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])


def print_measure(labels, measure, name):
    tmp = labels[measure.argsort()]
    tmp_idx = measure.argsort()
    print '\n', name,  '(Low to High) -'
    for idx in tmp_idx:
        print labels[idx], ':', measure[idx]


def print_results(acc, prec, rec, fsc):
    labels = np.array(['Airplane', 'Auto', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
    print '\nAccuracy =', acc
    print '\nOrder of precedences (Lowest to Highest) -'
    print 'F-Score :', str(labels[fscore.argsort()])
    print 'Precision :', str(labels[precision.argsort()])
    print 'Recall :', str(labels[recall.argsort()])
    
    print_measure(labels, fscore, 'F-Score')
    print_measure(labels, precision, 'Precision')
    print_measure(labels, recall, 'Recall')



if __name__ == "__main__":
    #caffe.set_mode_gpu()
    net = caffe.Net('conv.prototxt', 'cifar10_full_iter_60000.caffemodel.h5', caffe.TEST)
    #testModel('cifar10_full_iter_60000.caffemodel.h5')