# CudaDMASequential #

## Pattern Description ##

The CudaDMASequential transfer pattern is used for transferring a contiguous block of memory.  There are only two parameters required to characterize a sequential transfer pattern.
  * `ALIGNMENT` - the alignment of the block of memory (e.g. 4-, 8-, or 16-byte alignment)
  * `XFER_SIZE` - the size of the block of memory to be transferred in bytes

## Constructors ##

There are three constructors for the CudaDMASequential transfer pattern.  Different constructors all describe the same sequential transfer pattern, but allow for different parameters to be supplied as compile-time constants via template parameters.  Below are models for invoking the three constructors for CudaDMASequential.

```
/* Constructors for use with warp specialization */
cudaDMASequential<true/*specialized*/,ALIGNMENT,XFER_SIZE,DMA_THREADS>
  (dmaID, num_compute_threads, dma_threadIdx_start);

cudaDMASequential<true,ALIGNMENT,XFER_SIZE>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start);

cudaDMASequential<true,ALIGNMENT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start, xfer_size);

/* Constructors for use without warp specialization */
cudaDMASequential<false/*not specialized*/,ALIGNMENT,XFER_SIZE,TOTAL_THREADS>
  ();

cudaDMASequential<false,ALIGNMENT,XFER_SIZE>
  ();

cudaDMASequential<false,ALIGNMENT>
  (xfer_size);
```

The first constructor allows the user to supply the most number of compile-time constants as template parameters.  The user can specify the ALIGNMENT, XFER\_SIZE and the number of DMA\_THREADS as compile-time constants.  The second constructor keeps XFER\_SIZE as a compile time constant, while making the number of DMA threads a dynamic parameter.  The last constructor moves the transfer size parameter to being a dynamic parameter as well.  All other parameters are base parameters required by the CudaDMA [API](Interface.md).

For the non-warp-specialized constructors, the total threads parameter indicates to the CudaDMA object how many threads should be used to perform the transfer.  For the cases where total threads is not specified as a compile-time constant, we use `blockDim.x` as the number of threads to perform the transfer.

## Performance Considerations ##

Supplying as many parameters as possible as compile-time constants will contribute the most to achieving high performance with CudaDMASequential.  In addition to supplying compile-time constants, performance can also be achieved by aligning data to the largest byte-alignment possible.  16-byte alignment will perform better than 8-byte alignment, and 8-byte alignment will perform better than 4-byte alignment.