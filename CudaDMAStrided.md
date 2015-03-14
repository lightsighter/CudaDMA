# CudaDMAStrided #

## Pattern Description ##

The CudaDMAStrided pattern is used for transferring multiple blocks of data that are evenly spaced apart in memory.  Every block of memory must be of the same size.  We refer to these blocks of memory as elements.

The CudaDMAStrided pattern can be characterized by five different parameters.

  * `ALIGNMENT` - byte alignment of all elements (e.g. 4-,8-, or 16-byte aligned)
  * `BYTES_PER_ELMT` - the size of each element in bytes
  * `NUM_ELMTS` - the number of elements to be transfered
  * `src_stride` - the stride between the source elements in bytes
  * `dst_stride` - the stride between elements after they have been transferred in bytes

Both stride parameters are from the start of one element to the start of the next element.  The destination stride must be at least as large as `BYTES_PER_ELMT` or else the behavior of the transfer is undefined.  Also, the `ALIGNMENT` parameter must apply to be the source and destination locations of every element.  This implies that both `src_stride` and `dst_stride` must be divisible by the `ALIGNMENT` parameter.

## Constructors ##

The CudaDMAStrided pattern can be instantiated by a variety of constructors that allow different parameters to be passed as compile-time constants via template parameters.  In addition, there are are special constructors for the common case where the destination stride is equal to size of the elements.  Models for each of the different constructors for the CudaDMAStrided pattern can be seen below.

```
/* Constructors for use with warp specialization */
cudaDMAStrided<true/*specialized warps*/, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>
  (dmaID, num_compute_threads, dma_threadIdx_start, src_stride, dst_stride);
cudaDMAStrided<true, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>
  (dmaID, num_compute_threads, dma_threadIdx_start, src_stride); // dst_stride == BYTES_PER_ELMT

cudaDMAStrided<true, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS>
  (dmaID, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride, dst_stride);
cudaDMAStrided<true, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS>
  (dmaID, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride); // dst_stride == BYTES_PER_ELMT

cudaDMAStrided<true, ALIGNMENT, BYTES_PER_ELMT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride, dst_stride);
cudaDMAStrided<true, ALIGNMENT, BYTES_PER_ELMT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride); // dst_stride == BYTES_PER_ELMT

cudaDMAStrided<true, ALIGNMENT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   bytes_per_elmt, num_elmts, src_stride, dst_stride);
cudaDMAStrided<true, ALIGNMENT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   bytes_per_elmt, num_elmts, src_stride); // dst_stride == BYTES_PER_ELMT

/* Constructors for use without warp specialization */
cudaDMAStrided<false/*not specialized*/, ALIGNMENT, BYTES_PER_ELMT, TOTAL_THREADS, NUM_ELMTS>
  (src_stride, dst_stride);
cudaDMAStrided<false, ALIGNMENT, BYTES_PER_ELMT, TOTAL_THREADS, NUM_ELMTS>
  (src_stride); // dst_stride == BYTES_PER_ELMT

cudaDMAStrided<false, ALIGNMENT, BYTES_PER_ELMT, TOTAL_THREADS>
  (num_elmts, src_stride, dst_stride);
cudaDMAStrided<false, ALIGNMENT, BYTES_PER_ELMT, TOTAL_THREADS>
  (num_elmts, src_stride); // dst_stride == BYTES_PER_ELMT

cudaDMAStrided<false, ALIGNMENT, BYTES_PER_ELMT>
  (num_elmts, src_stride, dst_stride);
cudaDMAStrided<false, ALIGNMENT, BYTES_PER_ELMT>
  (num_elmts, src_stride); // dst_stride == BYTES_PER_ELMT

cudaDMAStrided<false, ALIGNMENT>
  (bytes_per_elmt, num_elmts, src_stride, dst_stride);
cudaDMAStrided<false, ALIGNMENT>
  (bytes_per_elmt, num_elmts, src_stride); // dst_stride == BYTES_PER_ELMT
```

Each constructor takes the base set of CudaDMA object arguments required by the CudaDMA [API](Interface.md).  In addition, each constructor takes the set of arguments required to fully characterize the CudaDMAStrided transfer pattern.

For the non-warp-specialized constructors, the total threads parameter describes the total number of threads to be used to perform the transfer.  In the cases where total threads is not supplied as a compile-time constant, the number of threads defaults to `blockDim.x`.  This is only true for the non-warp-specialized case.

## Performance Considerations ##
The highest possible performance for CudaDMAStrided pattern will be achieved when as many arguments as possible are specified as compile-time constants.  It's important to note that the number of DMA threads is more important as a compile-time constant than the number of elements to be transferred.  Higher performance can also be achieved by ensuring larger alignment of the elements.  16-byte alignment will perform better than 8-byte alignment, and 8-byte alignment will perform better than 4-byte alignment.