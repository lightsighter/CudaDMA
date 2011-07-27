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
import getopt
import os
import re
import math
import subprocess as sub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

### Globals
VERBOSE = False
SEED = 0

def main():
  parse_cmdline()
  outfile = "params_directed.h"

  do_compute = False
  double_buffer = False

  xs = []
  ys = []
  zs = []

  min_cnt = 1
  max_cnt = 64
  xticks = [1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
  #xticks = [1,8,16,24,32,40,48,56,64]
  #xticks = [1,8,16,32,48,64,128]
  min_sz = 16
  max_sz = 2*32*16*4
  yticks = [16, 32, 64, 32*4] + range(2*32*4*1, 2*32*16*4+1, 2*2*32*4*1)
  #yticks = [32*16*1, 32*16*2, 32*16*3, 32*16*4]

  for cnt in xticks:
    for sz in yticks:
        dma_sz = cnt*sz

        if dma_sz * 2 * 1 > 49152:
          xs.append(sz)
          ys.append(cnt)
          zs.append(0.0)
          print "Iter:", cnt, sz, dma_sz, "DNC"
          continue  

        cta_count = 14 if double_buffer else 28
        compute_warps = int(dma_sz/(4*32)) if int(dma_sz/(4*32)) < 16 else 16
        compute_warps = 1 if compute_warps == 0 else compute_warps
        #dma_warps = int(math.ceil(float(dma_sz)/float(64*32)))
        #dma_warps = 1 if dma_warps == 0 else dma_warps
        dma_warps = 1
        bytes_per_dma_thread = int(math.ceil(float(dma_sz)/float(32))) 
        mem_used = 2*dma_sz*cta_count*128/4
        use_small_el_opt = (sz <= 16*4)

        print "Iter:", cnt, sz, dma_sz, compute_warps, dma_warps, bytes_per_dma_thread, mem_used, "Using small el opt" if use_small_el_opt else "No opt"

        ########
        # Write new params_directed.h
        ########
        f = open(outfile,'w')
        if do_compute: f.write("#define DO_COMPUTE 1\n#define PRINT_ERRORS 1\n")
        if use_small_el_opt: 
            f.write("#define USE_SMALL_EL_OPT 1\n")
        if double_buffer:
            f.write("#define SAXPY_KERNEL saxpy_cudaDMA_doublebuffer\n")
        else:
            f.write("#define SAXPY_KERNEL saxpy_cudaDMA\n")
        f.write("#define CTA_COUNT " + str(cta_count) + "\n")
        f.write("#define COMPUTE_THREADS_PER_CTA 32 * " + str(compute_warps) + "\n")
        f.write("#define DMA_THREADS_PER_LD 32 * " + str(dma_warps) + "\n")
        f.write("#define NUM_ITERS 1024\n")
        #f.write("#define ALLOC_ITERS 128\n")
        f.write("#define BYTES_PER_DMA_THREAD "+ str(bytes_per_dma_thread) + "\n")
        f.write("#define DMA_SZ " + str(dma_sz) + "\n")
        f.write("#define EL_SZ " + str(sz) + "\n")
        f.close()

        ########
        # Run test
        ########
        p = sub.Popen('make clean ; make x86_64=1 keep=1 ; ../../bin/linux/release/saxpy_cudaDMA_strided', shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
        out = p.communicate()
        match = re.search('Bandwidth: (\d+\.\d+)', out[0])
        if match is not None: 
          xs.append(sz)
          ys.append(cnt)
          zs.append(float(match.group(1)))
        else: 
          print out[0]
          print out[1]
          xs.append(sz)
          ys.append(cnt)
          zs.append(0.0)
    
  import pickle
  output = open('data.pkl', 'wb')
  pickle.dump( (xs,ys,zs), output)
  output.close()
  print xs, ys, zs

  fig = plt.figure()
  ax = fig.add_subplot(111)

  xi = np.linspace(min_sz, max_sz, 1028)
  yi = np.linspace(min_cnt,max_cnt,1028)
  zi = griddata(np.array(xs),np.array(ys),np.array(zs),xi,yi)
  xi, yi = np.meshgrid(xi, yi)
  CS = plt.contourf(xi,yi,zi,cmap=plt.cm.bone)
  CS = plt.contourf(xi,yi,zi,cmap=plt.cm.bone)
  plt.colorbar()
  plt.clim(vmin=0)
  plt.scatter(xs,ys,marker='o',c='w',s=5,zorder=10, edgecolors=None)
  lbl_sz = 12
  ax.set_xlabel('Size of each element', size=lbl_sz)
  ax.set_ylabel('Number of elements', size=lbl_sz)

  from matplotlib.ticker import MultipleLocator
  xMajorLocator   = MultipleLocator(512)
  ax.xaxis.set_major_locator(xMajorLocator)
  yMajorLocator   = MultipleLocator(8)
  ax.yaxis.set_major_locator(yMajorLocator)

  plt.show()

### Command-line Parsing
def parse_cmdline():
  try:
    opts, args = getopt.getopt(sys.argv[1:], "hs:v",)
  except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"
    usage()
    sys.exit(2)
  if len(args)!=0:
    print "No arguments expected"
    usage()
    sys.exit(2)
  for o, a in opts:
    if o == "-v":
      global VERBOSE
      VERBOSE = True
    elif o in ("-h", "--help"):
      usage()
      sys.exit()
    elif o in ("-s"):
      global SEED
      SEED = a
    else:
      assert False, "unhandled option"
  return 

def usage():
    print """

    Summary:
    This script can generate random arguments useful for testing cudaDMA

    Outputs:
    params_random.h

    Usage:

    python param_gen.py [-s <seed>] [-v]

        -s: Random seed (default = 0)
        -v: Verbose
    """
  
if __name__ == "__main__":
    main()
