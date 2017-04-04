import cPickle as pickle
import urllib
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import idx2numpy
import copy 
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def sigmoid_weights(theta, x):
    p = -np.dot(theta, x)
    return 1 / (1 + np.exp(p))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def error_delta(theta, x, y):
    return np.sum(np.array([(sigmoid_weights(theta, x[i]) - y[i]) * x[i] for i in xrange(x.shape[0])]), axis=0)

def likelihood(theta, x, y):
    return np.sum([(y[i] * np.log(sigmoid_weights(theta, x[i]) + .001)) + ((1 - y[i]) * np.log(1 - sigmoid_weights(theta, x[i]) + .001)) for i in xrange(x.shape[0])])

def get_mnist_testset():
    x_train = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    x_train = np.array([x.flatten() for x in x_train])
    y_train = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    return x_train, y_train

def get_mnist_trainset():
    x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    x_train = np.array([x.flatten() for x in x_train])
    y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    return x_train, y_train

def merge_mnist_set(trainset, testset):
    """Merges two sets and returns a superset containing the two sets appended."""
    return np.append(trainset, testset, axis=0)

def softmax(arr, j):
    if arr[j] < 10:
        res = np.exp(arr[j]) / np.sum(np.exp(arr))
    else:
        arr -= arr.max()
        res = np.exp(arr[j]) / np.sum(np.exp(arr))
    return res

def computeZs(w, x):
    z = []
    z = sigmoid(np.dot(x, w.T))
    poly = PolynomialFeatures(1)
    z = poly.fit_transform(z)
    return np.array(z)

def computeYs(z, v):
    prods = np.dot(z, v.T)
    num = np.exp(prods)
    den = np.sum(num, axis=1)
    softmax_out = np.array([num[i] / den[i] for i in xrange(den.shape[0])])
    return softmax_out
    
def neural_likelihood(x_train, y_train, y_preds):
    k = len(set(y_train))
    m = x_train.shape[0]
    l = 1. * -np.sum([np.sum([np.log(y_preds[i][j] + .001) for j in xrange(k) if y_train[j] == j]) for i in xrange(m)])
    return l


def neural_predict_probs(x, w, v):
    z = computeZs(w, x)
    y = computeYs(z, v)
    return y

def neural_predict(x, w, v):
    preds_train = neural_predict_probs(x, w, v)
    return np.argmax(preds_train, axis=1)

def log_likelihood(y_hat, y):
  p = y_hat * y
  return -np.mean(np.log(p[p>0]))


def neural_train_test(x_train, y_train, x_test, y_test, hidden_units=64, momentum=.0, 
                      learn_rate=.05, reg=.01, degree=1, nfolds=4, limit=None):    
    batch_size = 128
    labels = 10
    epochs = 10

    train_ids = y_train < labels
    test_ids = y_test < labels

    x_train = x_train[train_ids]
    y_train = y_train[train_ids]
    x_test = x_test[test_ids]
    y_test = y_test[test_ids]

    n = x_train.shape[1]
    h = hidden_units
    k = len(set(y_train)) #numer of labels
    
    v = np.zeros([k, (1 + h)])
    #v = np.random.uniform(-.05, .05, [k, (1 + h)])
    v -= v.mean()
    prev_delta_v = np.zeros([k, (1 + h)])
    prev_delta_w = np.zeros([h, (1 + n)])
    w = np.ones([h, (1 + n)])
    #w = np.random.uniform(-.05, .05, [h, (1 + n)])
    w -= w.mean()

    poly = PolynomialFeatures(degree)
    x_train = poly.fit_transform(x_train)
    x_test = poly.fit_transform(x_test)
    z = []

    accs_iters = []; precs_iters = []; recs_iters = []; fscores_iters = []

    iter = 1
    y_oneHot = np.zeros([y_train.shape[0], k]) #Converting to one-hot encoding
    for i in xrange(y_oneHot.shape[0]):
        y_oneHot[i][y_train[i]] = 1

    prev_l = 0; l = 5

    train_limit = (x_train.shape[0] / batch_size) * batch_size

    x_train_all = copy.deepcopy(x_train)
    y_oneHot_all = copy.deepcopy(y_oneHot)
    y_train_all = copy.deepcopy(y_train)

    start_idx = 0
    epoch = 1
    ts = 0.
    avg_iter_time = 0.
    cnt = 1
    f = open('python_results.txt', 'wb')
    while epoch <= epochs:
        if (start_idx + 1) * batch_size > train_limit:
            start_idx = 0
            epoch += 1
            iter = 1

        st = timer()
        x_train = x_train_all[start_idx * batch_size:batch_size * (start_idx + 1)] / 255.
        y_oneHot = y_oneHot_all[start_idx * batch_size:batch_size * (start_idx + 1)]
        y_train = y_train_all[start_idx * batch_size:batch_size * (start_idx + 1)]
        start_idx += 1
            
        z = computeZs(w, x_train)
        y = computeYs(z, v)
            
        deriv_v = 1. * (y - y_oneHot) / batch_size
            
        delta_v = np.dot(deriv_v.T, z)
        deriv_w = np.dot(deriv_v, v)[:, 1:]

        deriv_z = (z * (1 - z))[:, 1:]
        delta_w = np.dot((deriv_z * deriv_w).T, x_train)
        
        delta_v = (delta_v * learn_rate) + (prev_delta_v * momentum)
        prev_delta_v = copy.deepcopy(delta_v)
        v -= (delta_v + (v * reg * learn_rate))
            
        delta_w = (delta_w * learn_rate) + (prev_delta_w * momentum)
        prev_delta_w = copy.deepcopy(delta_w)
        w -= (delta_w + (w * reg * learn_rate))
        et = timer()

        if cnt == 253:
          k = 0

        dur = et - st
        ts += dur
        if cnt == 1:
          avg_iter_time = dur
        else:
          avg_iter_time += dur
          avg_iter_time /= 2.

        preds = neural_predict_probs(x_train, w, v)
        wt_loss = (reg * .5 * ((v[:, 1:]**2).sum() + (w[:, 1:]**2).sum()))
        my_loss = log_likelihood(preds, y_oneHot)
        loss = my_loss + wt_loss

        f.write(str(ts) + ' ' + str(loss) + '\n')
        #y_pred = neural_predict(x_test[:batch_size], w, v)
        #acc = accuracy_score(y_test[:batch_size], y_pred)
        #print 'Batch', iter, 'Epoch =', epoch, 'Loss =', l, 'Accuracy =', acc
        
        print 'Batch', iter, 'Epoch =', epoch, 'Loss =', l, 'PYTHON_CPU', 'Avg iter time =', avg_iter_time

        prev_l = l
        iter += 1
        cnt += 1

    f.close()

if __name__ == "__main__":
    print '\nPerforming Neural Network experiments on MNIST dataset'
    x_mnist_train, y_mnist_train = get_mnist_trainset()
    x_mnist_test, y_mnist_test = get_mnist_testset()

    neural_train_test(x_mnist_train, y_mnist_train, x_mnist_test, y_mnist_test)