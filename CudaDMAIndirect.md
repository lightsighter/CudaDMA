# CudaDMAIndirect #

## Pattern Description ##

The CudaDMAIndirect transfer pattern performs arbitrary gather and scatter operations for a collection of data blocks in memory.  Each block of memory is referred to as an element.  The pattern assumes that every element has the same size.

The CudaDMAIndirect pattern can be characterized by four parameters.

  * `GATHER` - a boolean variable indicating whether the transfer is a gather or a scatter operation.  `true` indicates a gather operation while `false` indicates a scatter.
  * `ALIGNMENT` - the alignment of all of the elements to be transferred (e.g 4-, 8-, or 16-byte aligned)
  * `BYTES_PER_ELMT` - the size in bytes of each of the elements
  * `NUM_ELMTS` - the number of elements to be transferred

The call to `execute_dma` for the indirect pattern takes three arguments instead of the two used by most other CudaDMA patterns.

```
void execute_dma(const int *index_ptr, const void *src_ptr, void *dst_ptr);
```

The index pointer should point to an array of integers that describes the offsets for each of the elements specified as element offsets from source pointer (as opposed to byte offsets).  The index pointer array should contain at least `NUM_ELMTS` entries.  Additionally, this array must reside in memory visible to all of the DMA threads (either global or shared).  It cannot reside in local memory.

In the case of a gather operation, the elements are gathered from the offsets relative to the source pointer of the `execute_dma` call.  The elements are then written into the destination memory in a dense manner in the order of their offsets in the offset array.  For example, for each invocation of the `execute_dma` method the DMA threads will transfer element `i` from the location of `src_ptr+index_ptr[i]*BYTES_PER_ELMT` to the `dst_ptr+i*BYTES_PER_ELMT`.

Conversely, in the case of a scatter operation, elements will be read densely from the source location and then written into the corresponding destination location specified by the offset from the destination pointer for each element.

## Constructors ##

There are multiple constructors for CudaDMAIndirect that allow the user to specify variable numbers of parameters as compile-time constants via template parameters.  Each of the different constructors can be seen below.

```
/* Constructors for use with warp specialization */
cudaDMAIndirect<GATHER, true/*specialized*/,ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>
  (dmaID, num_compute_threads, dma_threadIdx_start);

cudaDMAIndirect<GATHER, true, ALIGNMENT, BYTES_PER_ELMT, DMA_THREADS>
  (dmaID, num_compute_threads, dma_threadIdx_start, num_elmts);

cudaDMAIndirect<GATHER, true, ALIGNMENT, BYTES_PER_ELMT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   num_elmts);

cudaDMAIndirect<GATHER, true, ALIGNMENT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   bytes_per_elmt, num_elmts);

/* Constructors for use without warp specialization */
cudaDMAIndirect<GATHER, false/*not specialized*/,ALIGNMENT, BYTES_PER_ELMT, TOTAL_THREADS, NUM_ELMTS>
  (void);

cudaDMAIndirect<GATHER, false, ALIGNMENT, BYTES_PER_ELMT, TOTAL_THREADS>
  (num_elmts);

cudaDMAIndirect<GATHER, false, ALIGNMENT, BYTES_PER_ELMT>
  (num_elmts);

cudaDMAIndirect<GATHER, false, ALIGNMENT>
  (bytes_per_elmt, num_elmts);
```

All parameters that are not directly related to the CudaDMAIndirect specification are required by the CudaDMA [API](Interface.md).

The total threads parameter for all the non-warp-specialized constructors indicates how many threads will be used by the CudaDMA object for performing the transfer.  When the total number of threads is not passed as a compile-time constant, then the total number of threads is inferred from `blockDim.x` value.

## Performance Considerations ##
The best performance for CudaDMAIndirect will be found when supplying as many compile-time constants as possible.  Note that knowing the number of DMA threads at compile-time is more important than knowing the number of elements to be transferred at compile-time.  Ensuring alignment of data to larger byte boundaries will also lead to higher performance.  16-byte alignment will perform better than 8-byte alignment, and 8-byte alignment will perform better than 4-byte alignment.

In addition, the ordering of offsets may impact performance.  The presence of L1 and L2 caches in the Fermi architecture may provide performance gains if elements on the same cache line are loaded by the DMA threads at the same time.  Therefore by grouping together offsets that are close in memory higher performance can be achieved.