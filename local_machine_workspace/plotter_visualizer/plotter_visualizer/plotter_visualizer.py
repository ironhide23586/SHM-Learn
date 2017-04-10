import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def process_file(fname):
  f = open(fname, 'rb')
  line = f.readline()
  times = []
  losses = []
  while line != '':
    if 'iter' in line:
      break
    vals = line.strip().split(' ')
    times.append(float(vals[0]))
    losses.append(float(vals[1]))
    line = f.readline()
  times = np.array(times)
  losses = np.array(losses)
  return times, losses


if __name__ == '__main__':
  sym = ['^', 'o', 'v', '<', '8', 's', '+', '*', 'x', 'd']
  res_folder_root = 'results' + os.sep + 'exp_14'
  files = glob.glob(res_folder_root + os.sep + '*.txt')
  results = {}
  plot_x_interval = 100
  for file in files:
    times, losses = process_file(file)
    key = file.split(os.sep)[-1].split('.')[0]
    results[key] = (times, losses)
  i = 0
  #batch_sizes = np.array([64, 128, 256, 512, 1024])
  #results = {'SHMLearn':[batch_sizes, np.array([.0003, .00048, .00047, .0006, .00098])], 'Caffe':[batch_sizes, np.array([.002, .003, .0023, .0026, .0038])]}
  #results = {'SHMLearn':[batch_sizes, np.array([2998, 2379, 2067, 1637, 580])], 'Caffe':[batch_sizes, np.array([465, 439, 415, 360, 267])]}
  for key in results.keys():
    x = results[key][0]
    y = results[key][1]
    #x = x[np.arange(0, x.shape[0], plot_x_interval)]
    #y = y[np.arange(0, y.shape[0], plot_x_interval)]
    #plt.xticks(range(x.shape[0]), x)
    #plt.plot(np.arange(len(x)), y, sym[i], label=key)
    plt.plot(x, y, sym[i], label=key)
    i += 1
  plt.xlabel('Time (in seconds)')
  #plt.xlabel('Batch Size')
  #plt.xticks(np.arange(len(batch_sizes)), batch_sizes)
  #plt.xlabel('Training Iteration')
  plt.ylabel('Training Loss (Log Likelihood)')
  #plt.ylabel('Time to complete one training iteration (in ms)');
  #plt.ylabel('Images processed per second');
  k = results.keys()
  #plt.title(k[0] + ' vs ' + k[1] + ' Performance plot')
  plt.legend(loc='upper right')
  plt.show()