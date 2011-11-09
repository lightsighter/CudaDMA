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
from matplotlib import cm

### Globals
VERBOSE = False
SEED = 0

def main():
  parse_cmdline()
  outfile = "params_directed.h"

  max_gbs = 133.36
  max_flops = 1

  compute_warps = 16
  #iters_per_compute_thread = [1,2,1024]
  #active_compute_warps =    [16,16,16]
  iters_per_compute_thread = [1,    1,   1,  1, 1, 1, 1, 1,2,4,8,16,32,64,128,256,512]
  active_compute_warps =     [0.125,0.25,0.5,1.,2.,4.,8.] + [float(compute_warps)]*10

  #Up to 3 allowed, more will run but be truncated from the plot
  sweeps = [ 
             ('saxpy_shmem', 'saxpy_shmem', True, False, False),
             ('saxpy_cudaDMA', 'saxpy_cudaDMA', True, False, True) 
           ]

  data = {}

  for series_name, kern_name, do_compute, double_buffer, use_cudadma in sweeps:
    xs = []
    ys_gbs = []
    ys_gflops = []

    for iters, active in zip(iters_per_compute_thread, active_compute_warps):
      cta_count = 14 if double_buffer else 28
      dma_warps = 4 if use_cudadma else compute_warps
      bytes_per_dma_thread = 64
      dma_sz = compute_warps*32*4

      compute_threads_per_cta = 32*compute_warps
      active_compute_threads_per_cta = int(32*active)

      ratio = float(dma_sz)/float(2*iters*active_compute_threads_per_cta)

      ########
      # Write new params_directed.h
      ########
      f = open(outfile,'w')
      if do_compute: f.write("#define DO_COMPUTE 1\n")
      #f.write("#define PRINT_ERRORS 1\n")
      f.write("#define SAXPY_KERNEL " + kern_name + "\n")
      f.write("#define CTA_COUNT " + str(cta_count) + "\n")
      f.write("#define COMPUTE_THREADS_PER_CTA " + str(compute_threads_per_cta) + "\n")
      f.write("#define ACTIVE_COMPUTE_THREADS_PER_CTA " + str(active_compute_threads_per_cta) + "\n")
      f.write("#define DMA_THREADS_PER_LD 32 * " + str(dma_warps) + "\n")
      f.write("#define NUM_ITERS 256\n")
      f.write("#define BYTES_PER_DMA_THREAD "+ str(bytes_per_dma_thread) + "\n")
      f.write("#define DMA_SZ " + str(dma_sz) + "\n")
      f.write("#define COMPUTE_THREAD_ITERS " + str(iters) + "\n")
      f.close()

      ########
      # Run test
      ########
      print "Iter running: ", ratio
      p = sub.Popen('make clean ; make x86_64=1 keep=1 ; ../../bin/linux/release/saxpy_cudaDMA_ratio', shell=True, stdout=sub.PIPE, stderr=sub.PIPE)
      out = p.communicate()
      match_b = re.search('Bandwidth: (\d+\.\d+)', out[0])
      match_p = re.search('Performance: (\d+\.\d+)', out[0])
      if match_b is not None and match_p is not None: 
        xs.append(ratio)
        ys_gbs.append(float(match_b.group(1)))
        ys_gflops.append(float(match_p.group(1)))
      else: 
        print out[0]
        print out[1]
        xs.append(ratio)
        ys_gbs.append(0.0)
        ys_gflops.append(0.0)
    data[series_name] = (xs, ys_gbs, ys_gflops)

  import pickle
  output = open('data.pkl', 'wb')
  pickle.dump( data, output)
  output.close()
  print data

  lbl_sz = 12
  styles = ['-','--',':']
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twinx()
  for name, vals, style in zip(data.keys(),data.values(),styles):
    xs, ys_gbs, ys_gflops = vals
    ax1.plot(xs,ys_gbs,'b'+style, label=name)
    ax2.plot(xs, ys_gflops, 'r'+style, label=name)

  ax1.set_xlabel('B/FLOP', size=lbl_sz)
  ax1.set_ylabel('GB/s', size=lbl_sz, color='b')
  for t1 in ax1.get_yticklabels():
    t1.set_color('b')
  ax2.set_ylabel('GFLOP/s', size=lbl_sz, color='r')
  for t1 in ax2.get_yticklabels():
    t1.set_color('r')
  ax1.set_xscale('log', basex=2)
  ax2.set_xscale('log', basex=2)
  ax1.invert_xaxis()
  ax2.invert_xaxis()
  ax1.set_ylim(0)
  ax2.set_ylim(0)
  leg = plt.legend(loc=7)
  for t in leg.get_texts():
    t.set_fontsize('small')
  for l in leg.get_lines():
    l.set_color('k')
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
