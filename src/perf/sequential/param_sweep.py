
import sys
import os
import math

def run_experiment(elmt_type,alignment,elmt_size,specialized,num_dma_warps,
                    cta_per_sm,buffering,loop_iters,out_file):
    print "Experiment "+str(elmt_type)+" "+str(alignment)+" "+str(elmt_size)+" "+str(specialized)+" "+ \
            str(num_dma_warps)+" "+str(cta_per_sm)+" "+str(buffering)+" "+str(loop_iters)+" "+str(out_file)
    f = open("params_directed.h","w")
    f.write("#define PARAM_ELMT_TYPE "+str(elmt_type)+"\n")
    f.write("#define PARAM_ALIGNMENT "+str(alignment)+"\n")
    f.write("#define PARAM_ELMT_SIZE "+str(elmt_size)+"\n")
    if (specialized):
        f.write("#define PARAM_SPECIALIZED true\n")
    else:
        f.write("#define PARAM_SPECIALIZED false\n")
    f.write("#define PARAM_DMA_WARPS "+str(num_dma_warps)+"\n")
    f.write("#define PARAM_CTA_PER_SM "+str(cta_per_sm)+"\n")
    f.write("#define PARAM_BUFFERING "+str(buffering)+"\n")
    f.write("#define PARAM_LOOP_ITERS "+str(loop_iters)+"\n")
    f.close()
    os.system('make clean; make')
    result = os.system('./perf_test >> '+str(out_file))
    if result:
        assert(False)

PARAM_ELMT_TYPE="float"
PARAM_ALIGNMENT=16
# Element size
elmt_size_start=32
elmt_size_stop=16384
elmt_size_stride=2
# DMA WARPS
dma_warps_start=1
dma_warps_step=2
dma_warps_stop=8
# CTAs per SM total
cta_per_sm_start=1
cta_per_sm_stride=2
cta_per_sm_stop=64
# LOOP ITERS
loop_iters_start=1
loop_iters_stride=2
loop_iters_stop=16

def run_all_experiments(specialized,buffering,out_file):
    elmt_size=elmt_size_start
    while elmt_size<=elmt_size_stop:
        for warps in range(dma_warps_start,dma_warps_stop+dma_warps_step,dma_warps_step):
            ctas = cta_per_sm_start
            while ctas<=cta_per_sm_stop:
                loops = loop_iters_start
                while loops<=loop_iters_stop:
                    run_experiment(PARAM_ELMT_TYPE,PARAM_ALIGNMENT,elmt_size,specialized,
                                   warps,ctas,buffering,loops,out_file)
                    loops = loops * loop_iters_stride
                ctas = ctas * cta_per_sm_stride
        elmt_size = elmt_size * elmt_size_stride
            

def main():
    out_file = "results.txt"
    run_all_experiments(False,"\"none\"",out_file)
    run_all_experiments(True,"\"single\"",out_file)
    run_all_experiments(True,"\"double\"",out_file)
    run_all_experiments(True,"\"manual\"",out_file)


if __name__ == "__main__":
    main()
