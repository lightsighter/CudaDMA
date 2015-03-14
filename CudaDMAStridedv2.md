# CudaDMAStrided Version 2.0 #

CudaDMAStrided [Version 2.0](http://code.google.com/p/cudadma/wiki/CudaDMAv2) has the same semantics and is characterized by the same parameters as the original [CudaDMAStrided](http://code.google.com/p/cudadma/wiki/CudaDMAStrided).

## Constructors ##
Constructors support from 3-6 template parameters for the new CudaDMAStrided transfer pattern.  All constructors support the new option for specifying the number of `BYTES_PER_THREAD` for outstanding LDG loads.  The value of `BYTES_PER_THREAD` must be a multiple of `ALIGNMENT`.  By selecting `4*ALIGNMENT` the implementation will default to the Fermi implementation.

```
/* Constructors for use with warp specialization */
CudaDMAStrided<true/*specialized warps*/, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>
  (dmaID, num_compute_threads, dma_threadIdx_start, src_stride, dst_stride);
CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, DMA_THREADS, NUM_ELMTS>
  (dmaID, num_compute_threads, dma_threadIdx_start, src_stride); // dst_stride == src_stride

CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, DMA_THREADS>
  (dmaID, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride, dst_stride);
CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, DMA_THREADS>
  (dmaID, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride); // dst_stride == stride

CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride, dst_stride);
CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   num_elmts, src_stride); // dst_stride == src_stride

CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   bytes_per_elmt, num_elmts, src_stride, dst_stride);
CudaDMAStrided<true, ALIGNMENT, BYTES_PER_THREAD>
  (dmaID, num_dma_threads, num_compute_threads, dma_threadIdx_start,
   bytes_per_elmt, num_elmts, src_stride); // dst_stride == src_stride

/* Constructors for use without warp specialization */
CudaDMAStrided<false/*not specialized*/, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, TOTAL_THREADS, NUM_ELMTS>
  (src_stride, dst_stride, dma_threadIdx_start/*optional*/);
CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, TOTAL_THREADS, NUM_ELMTS>
  (src_stride); // dst_stride == src_stride

CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, TOTAL_THREADS>
  (num_elmts, src_stride, dst_stride, dma_threadIdx_start/*optional*/);
CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT, TOTAL_THREADS>
  (num_elmts, src_stride); // dst_stride == src_stride

CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT>
  (num_elmts, src_stride, dst_stride, num_dma_threads/*optional*/, dma_threadIdx_start/*optional*/);
CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD, BYTES_PER_ELMT>
  (num_elmts, src_stride); // dst_stride == src_stride

CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD>
  (bytes_per_elmt, num_elmts, src_stride, dst_stride, num_dma_threads/*optional*/, dma_threadIdx_start/*optional*/);
CudaDMAStrided<false, ALIGNMENT, BYTES_PER_THREAD>
  (bytes_per_elmt, num_elmts, src_stride); // dst_stride == src_stride
```

Unlike previous versions of CudaDMA, the non-warp-specialized implementations also allow you to specify that a subset of the available warps should be used.  These are optional parameters.  Not specifying them will default to using all the threads in a threadblock for the transfer.

## Transfer Functions ##
CudaDMAStrided supports the following transfer functions.

```
class CudaDMAStrided {
public:
  // One-Phase Versions
  void execute_dma(const void *src_ptr, void *dst_ptr);

  template<bool GLOBAL_LOAD>
  void execute_dma(const void *src_ptr, void *dst_ptr);

  template<bool GLOBAL_LOAD, CudaDMALoadQualifier LOAD_QUAL, CudaDMAStoreQual STORE_QUAL>
  void execute_dma(const void *src_ptr, void *dst_ptr);

  // Two-Phase Versions
  void start_xfer_async(const void *src_ptr);

  template<bool GLOBAL_LOAD>
  void start_xfer_async(const void *src_ptr);

  template<bool GLOBAL_LOAD, CudaDMALoadQualifier LOAD_QUAL, CudaDMAStoreQual STORE_QUAL>
  void start_xfer_async(const void *src_ptr);

  void wait_xfer_finish(void *dst_ptr);

  template<bool GLOBAL_LOAD>
  void wait_xfer_finish(void *dst_ptr);

  template<bool GLOBAL_LOAD, CudaDMALoadQualifier LOAD_QUAL, CudaDMAStoreQual STORE_QUAL>
  void wait_xfer_finish(void *dst_ptr);
};
```

## Diagnostic Functions ##
CudaDMAStrided implements the following host-side diagnostic function.  It should be invoked with no template parameters.

```
template<...>
class CudaDMAStrided
{
public:
  __host__
  static void diagnose(int alignment, int bytes_per_thread, int bytes_per_elmt,
   int num_dma_threads, int num_elements, bool fully_templated, bool verbose = false);
};

// Example invocation
CudaDMAStrided<>::(/*arguments*/);
```