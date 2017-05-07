try:
  import caffe
except:
  l = 0
import idx2numpy
import numpy as np
import copy
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
import glob
import cPickle as pickle
import cv2

def cv22bgr(img_cv2):
  b, g, r = [img_cv2[:, :, i] for i in xrange(3)]
  return np.array([b, g, r])

def bgr2cv2(img_bgr):
  ret = np.array([img_bgr[:, i, :].T for i in xrange(img_bgr.shape[1])])
  return ret.astype(np.uint8)

def rgb2cv2(img_rgb):
  ret = np.array([img_rgb[:, i, :].T for i in xrange(img_rgb.shape[1])])
  return ret.astype(np.uint8)

def lin2bgr(img_linear, channels=3):
  lim = img_linear.shape[0] / channels
  side = np.sqrt(lim).astype(np.int)
  chnls = [img_linear[i*lim:(i + 1)*lim].reshape([side, side]) for i in xrange(channels)]
  img = np.zeros([channels, side, side])
  for c in xrange(channels):
    img[c, :, :] = chnls[channels - c - 1]
  return img

def lin2rgb(img_linear, channels=3):
  lim = img_linear.shape[0] / channels
  side = np.sqrt(lim).astype(np.int)
  chnls = [img_linear[i*lim:(i + 1)*lim].reshape([side, side]) for i in xrange(channels)]
  img = np.zeros([channels, side, side])
  for c in xrange(channels):
    img[c, :, :] = chnls[c]
  return img

def bgr2lin(img_bgr):
  return np.hstack([img_bgr[2-i].flatten() for i in xrange(img_bgr.shape[0])])

def show(img_cv2, res=(700, 700)):
  cv2.imshow('image', cv2.resize(img_cv2, res))
  cv2.waitKey()

#-----------------------------------------------------------------------#
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
#-----------------------------------------------------------------------#

def read_cifar10_worker(file_list):
  x_data = []
  y_data = []
  for data_file in file_list:
    read_data = pickle.load(open(data_file, 'rb'))
    x_data.append(read_data['data'])
    y_data.append(read_data['labels'])
  x_lin = np.vstack(x_data)
  y = np.hstack(y_data)
  x = np.array([lin2rgb(x_e) for x_e in x_lin])
  return x, y

def get_cifar10_trainset():
  train_files = glob.glob('cifar-10-batches-py/data*')
  return read_cifar10_worker(train_files)

def get_cifar10_testset():
  test_files = glob.glob('cifar-10-batches-py/test*')
  return read_cifar10_worker(test_files)

if __name__ == "__main__":
  #x_train, y_train = get_mnist_trainset()
  #x_test, y_test = get_mnist_testset()

  x_train, y_train = get_cifar10_trainset()
  x_test, y_test = get_cifar10_testset()

  caffe.set_mode_gpu()
  batch_size = 128
  labels = 10
  epochs = 10
  learn_rate = .05
  reg = .01
  train_ids = y_train < labels
  test_ids = y_test < labels
  x_train = x_train[train_ids]
  y_train = y_train[train_ids]
  x_test = x_test[test_ids]
  y_test = y_test[test_ids]
  #net = caffe.Net('net.prototxt', caffe.TRAIN)
  solver = caffe.SGDSolver('solver.prototxt')

  #solver.net.params['ip0'][0].data[:] = np.random.uniform(-.05, .05, solver.net.params['ip0'][0].data.shape)
  #solver.net.params['ip1'][0].data[:] = np.random.uniform(-.05, .05, solver.net.params['ip1'][0].data.shape)
  ##solver.net.params['ip0'][0].data[:] = np.ones(solver.net.params['ip0'][0].data.shape)
  ##solver.net.params['ip1'][0].data[:] = np.zeros(solver.net.params['ip1'][0].data.shape)

  #solver.net.params['ip0'][0].data[:] -= solver.net.params['ip0'][0].data.mean()
  #solver.net.params['ip1'][0].data[:] -= solver.net.params['ip1'][0].data.mean()
  #solver.net.params['ip0'][0].data[:] = np.ones(solver.net.params['ip0'][0].data.shape)
  #solver.net.params['ip1'][0].data[:] = np.zeros(solver.net.params['ip1'][0].data.shape)
  y_oneHot = np.zeros([y_train.shape[0], labels]) #Converting to one-hot encoding
  for i in xrange(y_oneHot.shape[0]):
    y_oneHot[i][y_train[i]] = 1
  train_limit = (x_train.shape[0] / batch_size) * batch_size
  x_train_all = copy.deepcopy(x_train)
  y_oneHot_all = copy.deepcopy(y_oneHot)
  y_train_all = copy.deepcopy(y_train)
  start_idx = 0
  epoch = 1
  iter = 1
  ts = 0.
  avg_iter_time = 0.
  cnt = 1
  f = open('caffe_results.txt', 'wb')
  while epoch <= epochs: 
    if (start_idx + 1) * batch_size > train_limit:
        start_idx = 0
        epoch += 1
        iter = 1
    x_train = x_train_all[start_idx * batch_size:batch_size * (start_idx + 1)] / 255.
    y_oneHot = y_oneHot_all[start_idx * batch_size:batch_size * (start_idx + 1)]
    y_train = y_train_all[start_idx * batch_size:batch_size * (start_idx + 1)]

    st = timer()
    solver.net.blobs['data'].data[:] = x_train
    solver.net.blobs['label'].data[:] = y_train
    l = solver.step(1)
    et = timer()
    dur = (et - st)
    ts += dur

    if cnt == 253:
      k = 0

    if cnt == 1:
      avg_iter_time = dur
    else:
      avg_iter_time += dur
      avg_iter_time /= 2.

    solver.net.forward()
    preds = solver.net.blobs['prob'].data
    solver.net.blobs['prob'].diff[:] = (preds - y_oneHot) / 10.
    p = y_oneHot * preds
    my_loss = np.mean(-np.log(p[p>0]))
    wt_loss = reg * .5 * np.sum([(solver.net.params[k][0].data**2).sum() + (solver.net.params[k][1].data**2).sum() for k in solver.net.params.keys()])
    
    loss = my_loss + wt_loss
    pred_labels = preds.argmax(axis=1)
    #print 'Batch', iter, 'Epoch =', epoch, 'Loss =', my_loss, 'CAFFE_GPU', 'Avg iter time =', avg_iter_time
    
    print("Batch %d, Epoch = %d, Loss = %f, Actual Loss = %f, Wt Loss = %f, CAFFE_CUDA_GPU Avg iter time = %f" % (iter, epoch, loss, my_loss, wt_loss, avg_iter_time))             

    f.write(str(ts) + ' ' + str(loss) + '\n')
    start_idx += 1
    iter += 1

    if ts >= 1:
      k = cnt

    cnt += 1

f.write('Avg iter time = ' + str(avg_iter_time) + ' seconds\n')
f.close()