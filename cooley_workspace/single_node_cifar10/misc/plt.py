import matplotlib.pyplot as plt
import numpy as np

f = open('shmlearn_results.txt', 'rb')
lines = f.readlines()
l = np.array([float(line.split(',')[3].split('=')[-1].strip()) for line in lines if 'nan' not in line])
l = l[l < 5]

plt.plot(range(len(l)), l)
plt.show()