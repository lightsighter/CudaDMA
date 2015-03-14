
# Best Practices #
There are several best practices that we use when creating CudaDMA objects.  These practices allow CudaDMA objects to achieve high performance.  They also can give the user some insight into the performance of the CudaDMA objects that have already been implemented.  If you're writing custom CudaDMA objects it's not necessary to abide by all of these recommendations, but they do tend to lead to CudaDMA objects that achieve high performance.

## Compile-Time Constants ##
In as many cases as possible, we allow the programmer to specify parameters to the CudaDMA object as compile-time constants via template parameters.  The `nvcc` compiler performs aggressive constant folding at compile-time.  Passing parameters as template arguments facilitates this process.

From a programmability standpoint, compile-time constants can make it more difficult to use CudaDMA objects.  As a result, we supply default values for template parameters that a programmer would never use and then specialize the templates for these cases to allow arguments to also be passed as runtime arguments to the constructors for CudaDMA objects.  By providing multiple versions of a CudaDMA object through template specialization, we give the programmer the option to chose whether to make arguments compile-time or runtime arguments.

## Loading at Maximum Alignment ##
The reason that we require the programmer to specify the alignment of the pointers passed to the `execute_dma` function is that it enables CudaDMA objects to issue loads for more data if the guaranteed alignment is larger.  For example, if a 16-byte alignment is guaranteed, then CudaDMA objects will [issue 16](https://code.google.com/p/cudadma/issues/detail?id=16)-byte vector loads (i.e. `float4` or `int4`) whenever possible in order to minimize the number of memory requests that have to be issued for an entire transfer.

## Outstanding Loads Per Thread ##
Another performance optimization that we make is to issue as many loads as possible from DMA threads without exceeding the number of load issue slots per thread.  By not exceeding the maximum number of outstanding loads per thread we avoid stalling DMA threads while still maximizing MLP.  We currently issue four loads per DMA thread per step of a transfer.

## Prefetching Into Registers ##
While the CudaDMA API indicates that the DMA threads must first wait for a call to `start_async_dma` by the compute threads before a transfer can begin, the DMA threads can actually start issuing loads prior to this call as long as they keep their values in registers before issuing their writes.  This facilitates even better overlapping of computation and memory accesses.

In some cases users may need strict ordering on memory operations that require that this prefetching is not permitted.  In these cases the user can do explicit synchronization using `wait_for_dma_start` and `finish_async_dma` calls around `execute_dma_no_sync` operations.

## Hoisting Pointer Arithmetic ##
When implementing CudaDMA objects, we also attempt to hoist as much pointer arithmetic into the constructor as possible.  By keeping these operations in the constructor we avoid placing them in the inner loop of the DMA threads as part of the `execute_dma` call.  To aid in this process, we often implement CudaDMA objects based on offsets from the `src_ptr` and `dst_ptr` passed to the `execute_dma` call rather than explicit pointers.

## Avoiding Memory Conflicts ##
The last optimization that we make when implementing CudaDMA objects is to avoid both bank conflicts in shared memory and replays due to a lack of coalescing in global memory accesses.  The CudaDMA API decouples a transfer from its implementation, which enables the implementor of CudaDMA objects to decide which loads are performed by which DMA threads.  We implement CudaDMA objects to have maximal coalescing when performing accesses to global memory and minimal bank conflicts when accessing shared memory.