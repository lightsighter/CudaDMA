
import sys
import os

MAX_ELMT_SIZE=8192 # in floats
MAX_NUM_ELMTS=32
MAX_DMA_WARPS=16

def run_experiment(alignment,offset,elem_size): #,num_elmts,dma_warps):
    f = open("params_directed.h",'w')
    f.write("#define PARAM_ALIGNMENT "+str(alignment)+"\n")
    f.write("#define PARAM_OFFSET "+str(offset)+"\n")
    f.write("#define PARAM_ELMT_SIZE "+str(elem_size*4)+"\n")
    #f.write("#define PARAM_NUM_ELMTS "+str(num_elmts)+"\n")
    #f.write("#define PARAM_DMA_THREADS "+str(dma_warps*32)+"\n")
    f.close()
    os.system('make clean; make')
    result = os.system('./test_strided')
    if not(result):
        assert(False)
    #if result:
    #    print "Experiment: ALIGNMENT-"+str(alignment)+" OFFSET-"+str(offset)+" ELMT_SIZE-"+str(4*elem_size)+" NUM_ELMTS-"+str(num_elmts)+" DMA_WARPS-"+str(dma_warps)+" Result: SUCCESS"
    #else:
    #    print "Experiment: ALIGNMENT-"+str(alignment)+" OFFSET-"+str(offset)+" ELMT_SIZE-"+str(4*elem_size)+" NUM_ELMTS-"+str(num_elmts)+" DMA_WARPS-"+str(dma_warps)+" Result: FAILURE"
    #    assert(False)
 

def run_all_experiments(alignment,offset):
    for elem_size in range(1,MAX_ELMT_SIZE):
        run_experiment(alignment,offset,elem_size)
    #for elem_size in range(1,MAX_ELMT_SIZE):
    #    for num_elems in range(1,MAX_NUM_ELMTS):
    #	    for warps in range(1,MAX_DMA_WARPS):
    #            run_experiment(alignment,offset,elem_size,num_elems,warps)

def main():
    run_all_experiments(16,0)
    run_all_experiments(8,0)
    run_all_experiments(8,2)
    run_all_experiments(4,0)
    run_all_experiments(4,1)
    run_all_experiments(4,2)
    run_all_experiments(4,3)

if __name__ == "__main__":
    main()
