import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool

class Neuron:

  w = None
  x = None
  y = None
  loss = None
  lr = None
  w_grad = None
  activation = "relu"

  def __init__(self, w=0.5, learn_rate=.5):
    self.w = w
    self.lr = learn_rate

  def fwd(self, inp):
    self.x = inp
    self.y = self.act(self.w * inp)
    return self.y

  def act(self, inp):
    if self.activation == "sigmoid":
      return 1. / (1. + np.exp(-inp))
    elif self.activation == "relu":
      return np.max(0, inp)

  def loss_func(self, inp, op):
    if op == 0:
      self.loss = inp
    elif op == 1:
      self.loss = 1 - self.fwd(inp)
    return self.loss

  def weight_grad(self, inp, op):
    y = self.fwd(inp)
    dy = 0
    if self.activation == "sigmoid":
      dy = y * (1. - y)
    elif self.activation == "relu":
      if y > 0:
        dy = 1
      else:
        dy = 0
    if op == 1:
      dy *= -1.
    self.w_grad = dy * self.x * self.lr
    return self.w_grad

  def apply_grad(self, g):
    self.w -= g

  def train(self, x, y):
    self.weight_grad(x, y)
    self.w -= self.w_grad
    return self.loss_func(x, y)

def gen_set(sz):
  x = np.random.rand(sz)
  y = copy.deepcopy(x)
  y[y >= .5] = 1.
  y[y < .5] = 0.
  return x, y

def train_set(neuron, data, iters):
  losses = []
  epoch_size = data[0].shape[0]
  for i in xrange(iters):
    l = neuron.train(data[0][i % epoch_size], data[1][i % epoch_size])
    losses.append(l)
  return losses

def test_neuron(neuron, sz=100):
  x, y = gen_set(sz)
  preds = np.array([neuron.fwd(x_e) for x_e in x])
  preds[preds >= .5] = 1.
  preds[preds < .5] = 0.
  return 1. * (preds * y).sum() / sz

def train_neuron(neuron, iters=100, sz=100):
  d = gen_set(sz)
  l = train_set(neuron, d, iters)
  return l

def partition(arr, parts):
  ans = []
  segment_size = int(math.ceil(len(arr) / parts))
  for i in xrange(parts):
    if i < (parts - 1):
      ans.append(arr[i * segment_size : (i * segment_size) + segment_size])
    else:
      ans.append(arr[i * segment_size:])
  return ans


def avg_models_diff(neurons):
  w_mean = np.mean([neuron.w for neuron in neurons])
  return Neuron(w_mean)

def train_dist_diff(neurons, data, steps):
  num_models = len(neurons)
  data_parts = (partition(data[0], num_models), partition(data[1], num_models))
  minibatch_size = len(data_parts[0][0])
  trained_models = []
  losses_diff = []
  for step in xrange(steps):
    x_models = [data_parts[0][i][step % minibatch_size] for i in xrange(num_models)]
    y_models = [data_parts[1][i][step % minibatch_size] for i in xrange(num_models)]
    for i in xrange(num_models):
      neurons[i].train(x_models[i], y_models[i])
    n = avg_models_diff(neurons)
    trained_models.append(copy.deepcopy(n))
    for i in xrange(num_models):
      neurons[i].w = n.w
    losses_diff.append(np.mean([n.loss_func(x_models[i], y_models[i]) for i in xrange(num_models)]))
  return trained_models, losses_diff

def train_dist_eq(neurons, data, steps):
  num_models = len(neurons)
  data_parts = (partition(data[0], num_models), partition(data[1], num_models))
  minibatch_size = len(data_parts[0][0])
  trained_models = []
  losses_eq = []
  ret_neuron = copy.deepcopy(neurons[0])
  for step in xrange(steps):
    x_models = [data_parts[0][i][step % minibatch_size] for i in xrange(num_models)]
    y_models = [data_parts[1][i][step % minibatch_size] for i in xrange(num_models)]
    grads = []
    for i in xrange(num_models):
      grads.append(neurons[i].weight_grad(x_models[i], y_models[i]))
    ret_neuron.apply_grad(np.mean(grads))
    for i in xrange(num_models):
      neurons[i].w = ret_neuron.w
    trained_models.append(copy.deepcopy(ret_neuron))
    losses_eq.append(np.mean([ret_neuron.loss_func(x_models[i], y_models[i]) for i in xrange(num_models)]))
  return trained_models, losses_eq


def train_dist(num_models, data_sz=400, steps=10000):
  init_wts_diff = np.random.rand(num_models)
  #init_wts_eq = np.mean(init_wts_diff)
  init_wts_eq = np.random.rand()
  m_diff = [Neuron(w) for w in init_wts_diff]
  m_eq = [Neuron(init_wts_eq)] * num_models
  x, y = gen_set(data_sz)
  n_diff, w_avg_losses = train_dist_diff(m_diff, (x, y), steps)
  n_eq, grad_avg_losses = train_dist_eq(m_eq, (x, y), steps)
  return [n_diff, w_avg_losses], [n_eq, grad_avg_losses]

def train_dist_only_grad(num_models, data_sz=400, steps=10000):
  init_wts_diff = np.random.rand(num_models)
  #init_wts_eq = np.mean(init_wts_diff)
  init_wts_eq = np.random.rand()
  m_diff = [Neuron(w) for w in init_wts_diff]
  m_eq = [Neuron(init_wts_eq)] * num_models
  x, y = gen_set(data_sz)
  n_eq, grad_avg_losses = train_dist_eq(m_eq, (x, y), steps)
  return n_eq, grad_avg_losses



if __name__ == "__main__":

  workers = [2, 4, 8, 16]
  sym = ['+', '.', '-', '*']
  p = Pool(8)

  loss_graphs = p.map(train_dist_only_grad, workers)

  for i in xrange(len(workers)):
    plt.plot(np.arange(1, len(loss_graphs[i][1]) + 1), loss_graphs[i][1], '.', label=str(workers[i]) + ' workers')


  #w_avg, grad_avg = train_dist(8)
  #loss_graphs = {"Weight Averaged": w_avg[1], "Gradient Averaged": grad_avg[1]}
  
  #sym = ['+', '.']
  #i = 0
  #for k in loss_graphs.keys():
  #  plt.plot(np.arange(1, len(loss_graphs[k]) + 1), loss_graphs[k], sym[i], label=k)
  #  i += 1

  plt.xlabel('Steps per worker')
  plt.ylabel('Training Loss')

  plt.legend()
  plt.show()