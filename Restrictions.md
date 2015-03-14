# Restrictions #

## One Dimensional Threadblocks ##

The CudaDMA interface currently assumes that all threadblocks are one dimensional in the x-dimension (i.e. `blockDim.y==1` and `blockDim.z==1`).  This restriction is because it is easier to separate DMA warps from compute warps in only one dimension.  Note that this doesn't preclude launching multi-dimensional grids of threadblocks, only that each threadblock enumerate its threads in the x-dimension.

## Explicit DMA Warps ##

Explicit DMA warps are only supported on devices of compute capability 2.0.  The reason for this is that the fine-grained synchronization primitives used for supporting CudaDMA are only implemented on Fermi devices.

## Barrier Granularity ##
Barriers on current CUDA devices only support performing barriers at warp granularity.  Therefore each CudaDMA object must be given a number of threads that is an integer multiple of `warpSize` (currently 32 on all CUDA devices).

## Use of `__syncthreads` ##
The `__syncthreads` primitive is not explicitly invalidated by using CudaDMA code, but programmers should be cautious in combining the two.  Each CudaDMA object is given an ID that specifies the two named barriers used by that particular object (as described in the CudaDMA [API](Interface.md)).  If a CudaDMA object is given ID 0, then it will use named barriers 0 and 1.  However, the `__syncthreads` primitive implicitly uses named barrier 0 and could result in barrier aliasing and undefined behavior.  Therefore when using `__syncthreads` we recommend numbering CudaDMA objects starting at ID 1.  Note also that it is still possible to cause deadlock by using `__syncthreads` in conjunction with CudaDMA.

It is still possible to simulate the effect of `__syncthreads` in either a compute or a DMA warp section of code using an internal CudaDMA function.  Included in the `cudaDMA.h` header file is the following function

```
__device__ void ptx_cudaDMA_barrier_blocking(const int name, const int num_threads);
```

This device allows the programmer to barrier a subset of the threads in a threadblock.  The first parameter is the number indicating which named barrier should be used.  Ideally this barrier should be different than any of the barriers used by the CudaDMA objects to avoid aliasing and deadlocks.  However, a deep understanding of the CudaDMA barrier protocol may allow re-use of barriers.  The second parameter indicates the number of threads that have to hit the barrier before it is completed.  This should be the number of DMA threads or compute threads in the if-else block of code.

It should be noted that the primarily use of this call is when shared memory buffers are used for multiple purposes by the compute threads.  We have yet to find a reason to barrier DMA threads independent of the CudaDMA functions that make use of named barriers.

## Maximum Named Barriers ##

Every streaming multiprocessor (SM) on a CUDA device has fixed resources such as the amount of shared memory and registers that it contains.  The number of named barriers on a device is also a constrained resource.  There are up to 16 named barriers on an SM.  Therefore the maximum number of CudaDMA objects in a kernel is 8 since each CudaDMA object requires two named barriers.

In addition to this restriction, note that it's possible for named barrier usage to reduce the maximum occupancy of a kernel.  Using many CudaDMA objects in a kernel and consequently multiple named barriers can lower the maximum number of threadblocks placed on an SM.

## Loop Unrolling ##

The `nvcc` compiler currently supports pragmas that enable loop unrolling.  However, this feature is currently not supported when the loop contains inlined PTX code, even code inside of function calls in loops (i.e. CudaDMA synchronization calls).  We have not noticed any significant performance degradation from this, but are working with the CUDA compiler team to address it.