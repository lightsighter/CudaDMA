# CudaDMA Object API #

Every CudaDMA object must implement the base CudaDMA object API.  The API consists of a constructor and collection of methods on the CudaDMA object that enable the programmer to

  * create instances of the CudaDMA object
  * separate DMA warps from compute warps
  * synchronize between DMA and compute warps
  * repeatedly perform data transfer operations

The CudaDMA API is shown below.

```
class cudaDMA
{
public:
  // Base Constructor
  __device__ cudaDMA (const int dmaID,
                      const int num_dma_threads,
                      const int num_compute_threads,
                      const int dma_threadIdx_start);
public:
  __device__ bool owns_this_thread();
public:
  // Compute thread synchronization functions
  __device__ void start_async_dma();
  __device__ void wait_for_dma_start();
public:
  // DMA thread synchronization functions
  __device__ void wait_for_dma_finish();
  __device__ void finish_async_dma();
public:
  __device__ void execute_dma(void *src_ptr,
                              void *dst_ptr) const;
  __device__ void execute_dma_no_sync(void *src_ptr,
                              void *dst_ptr) const;
};
```

## Constructor ##
The constructor for a CudaDMA object provides information to the implementation of the object that enables it to function correctly.  These fields must be supplied as a part of all CudaDMA objects.  The fields supplied here only matter for the case where warp specialization is being used as described by the CudaDMA [programming model](ProgrammingModel.md).

  * `dmaID` - This must be a unique integer starting from 0 and counting up by 1 for each CudaDMA object in the SAME kernel.  CudaDMA objects in different kernels may use the same ID.  The choice of the DMA ID also has implications for the `__syncthreads` intrinsic explained in the section on [restrictions](Restrictions.md).
  * `num_dma_threads` - Indicate the number of DMA threads that this object will be responsible for managing.  This value must be a multiple of the `warpSize` parameter (32 threads on current NVIDIA hardware).
  * `num_compute_threads` - The number of compute threads active in the kernel with which the CudaDMA object will need to synchronize.  This value must also be a multiple of `warpSize`.
  * `dma_threadIdx_start` - The `x` index of the first DMA thread that the object will be responsible for managing.  We assume that all threads in a threadblock are laid out in the x-dimension only.

## Managing DMA and Compute Warps ##
There are several different function calls for managing DMA threads and allowing them to synchronize with compute threads.  We describe each of them in turn.

  * `owns_this_thread` - This function will return true if the thread is one of the DMA threads managed by the specified CudaDMA object.
  * `start_async_dma` - A non-blocking function that can be performed by the compute threads.  The call indicates to the DMA threads that the buffer is empty and should now be filled.  The asynchronous nature of this call allows compute threads to perform additional computation while waiting for the transfer to complete.
  * `wait_for_dma_finish` - A blocking function called from the compute threads to wait for a transfer into a buffer to complete.
  * `wait_for_dma_start` - A blocking function called from the DMA threads that will wait until the compute threads indicate that the buffer is empty and ready to be filled.
  * `finish_async_dma` - A non-blocking function that allows DMA threads to continue performing useful work even after completing a DMA operation.

## Performing Transfers ##
There are two different function calls for performing data transfers.  They both take the same arguments and will perform the same operation.  The difference is whether or not synchronization is performed by the DMA threads when executing the DMA operation.

  * `execute_dma` - This operation will perform the transfer with the expectation that there are separate DMA threads that need to synchronize with the compute threads.  The calls to `wait_for_dma_start` and `finish_async_dma` are internalized in the code for this function as they enable performance optimizations such as prefetching data into registers.  If `execute_dma` is used there is no need to call the DMA synchronizing functions.
  * `execute_dma_no_sync` - This function performs the same operation as `execute_dma` but does not contain any synchronization calls.  This is useful if the program is not using separate DMA threads and does not require fine grained synchronization.  Note that `__syncthreads` call is still necessary to synchronize all threads to wait for the DMA operation to finish in this case.