#/usr/bin/python

###########################################################################
#  Copyright 2010 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###########################################################################

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
#from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

### Globals
VERBOSE = False
SEED = 0

def main():

  do_compute = False
  double_buffer = False
  use_opts = False

  xs = []
  ys = []
  zs = []

  min_cnt = 1
  max_cnt = 64
  xticks = [1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
  min_sz = 32*4
  max_sz = 2*32*16*4
  yticks = [32*4] + range(2*32*4*1, 2*32*16*4+1, 2*32*4*1)
  #xticks = range(1,5)
  #yticks = [32*16*1, 32*16*2, 32*16*3, 32*16*4]
    
  import pickle
  f = open('data.pkl', 'r')
  data = pickle.load(f)
  xs = data[0]
  ys = data[1]
  zs = data[2]

  fig = plt.figure()
  print xs, ys, zs
  ax = fig.add_subplot(111)

  xi = np.linspace(min_sz, max_sz, 1028)
  yi = np.linspace(min_cnt,max_cnt,1028)
  zi = griddata(np.array(xs),np.array(ys),np.array(zs),xi,yi)
  #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
  xi, yi = np.meshgrid(xi, yi)
  CS = plt.contourf(xi,yi,zi,[0,25,50,75,100,125,150],cmap=plt.cm.bone,extend='min')
  CS = plt.contourf(xi,yi,zi,[0,25,50,75,100,125,150],cmap=plt.cm.bone,extend='min')
  #CS = plt.contourf(xi,yi,zi,cmap=plt.cm.bone,extend='min')
  #CS = plt.contourf(xi,yi,zi,cmap=plt.cm.bone,extend='min')
  #plt.clim(vmin=-1)
  plt.colorbar()
  plt.scatter(xs,ys,marker='o',c='w',s=5,zorder=10, edgecolors=None)
  lbl_sz = 12
  ax.set_xlabel('Size of each element (bytes)', size=lbl_sz)
  ax.set_ylabel('Number of elements', size=lbl_sz)

  from matplotlib.ticker import MultipleLocator
  xMajorLocator   = MultipleLocator(512)
  ax.xaxis.set_major_locator(xMajorLocator)
  yMajorLocator   = MultipleLocator(8)
  ax.yaxis.set_major_locator(yMajorLocator)

  plt.show()

if __name__ == "__main__":
    main()
