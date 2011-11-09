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

### Globals
VERBOSE = False
SEED = 0

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

### Main Program ###

def main():
  parse_cmdline()
  outfile = "params_directed.h"

  #warp_counts = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
  warp_counts = [4]
  for i in warp_counts:

    ########
    # Write new params_directed.h
    ########
    f = open(outfile,'w')
    f.write("#define SAXPY_KERNEL saxpy_cudaDMA\n")
    f.write("#define CTA_COUNT 28\n")
    #f.write("#define SAXPY_KERNEL saxpy_cudaDMA_doublebuffer\n")
    #f.write("#define CTA_COUNT 14\n")
    f.write("#define COMPUTE_THREADS_PER_CTA 32 * 16 \n")
    f.write("#define DMA_THREADS_PER_LD 32 * " + str(i) + "\n")
    f.write("#define NUM_ITERS 2048\n")
    f.write("#define BYTES_PER_DMA_THREAD 64\n")
    f.write("#define DMA_SZ 4 * COMPUTE_THREADS_PER_CTA\n")
    f.close()

    ########
    # Run test
    ########
    os.system('make clean ; make x86_64=1 keep=1 ; ../../bin/linux/release/saxpy_cudaDMA')
  
if __name__ == "__main__":
    main()
