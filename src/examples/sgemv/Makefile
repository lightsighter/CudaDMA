
all: vec_single 

#DEBUG=1
#PROFILING=1
#SASS=1

CU_FLAGS= -O2 -arch=compute_20 -code=sm_20 --ptxas-options="-v" -keep # -abi=no" 

ifdef DEBUG
CU_FLAGS+= -g -DDEBUG
endif

ifdef PROFILING
CU_FLAGS+= -DPROFILING
endif

vec_single:
	nvcc -o sgemv $(CU_FLAGS) -DVEC_SINGLE -maxrregcount 32 sgemv_dma.cu

vec_double:
	nvcc -o sgemv $(CU_FLAGS) -DVEC_DOUBLE -maxrregcount 63 sgemv_dma.cu

vec_manual:
	nvcc -o sgemv $(CU_FLAGS) -DVEC_MANUAL -maxrregcount 32 sgemv_dma.cu

both_single:
	nvcc -o sgemv $(CU_FLAGS) -DBOTH_SINGLE -maxrregcount 63 sgemv_dma.cu

both_double:
	nvcc -o sgemv $(CU_FLAGS) -DBOTH_DOUBLE -maxrregcount 63 sgemv_dma.cu

both_manual:
	nvcc -o sgemv $(CU_FLAGS) -DBOTH_MANUAL -maxrregcount 57 sgemv_dma.cu

mat_single:
	nvcc -o sgemv $(CU_FLAGS) -DMAT_SINGLE -maxrregcount 63 sgemv_dma.cu

clean:
	rm -f *.o sgemv
